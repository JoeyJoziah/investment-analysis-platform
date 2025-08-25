"""
Comprehensive Injection Prevention System
Protects against SQL injection, XSS, CSRF, and other injection attacks
"""

import re
import html
import json
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union, Callable, Pattern
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
from urllib.parse import quote, unquote, urlparse

# Database imports
import sqlalchemy
from sqlalchemy import text, inspect
from sqlalchemy.sql import sqltypes
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

# FastAPI imports
from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

# Security libraries
import bleach
from bleach.sanitiser import Cleaner
from bleach.linkifier import Linker
import html5lib

logger = logging.getLogger(__name__)


class AttackType(str, Enum):
    """Types of injection attacks"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    LDAP_INJECTION = "ldap_injection"
    XPATH_INJECTION = "xpath_injection"
    COMMAND_INJECTION = "command_injection"
    TEMPLATE_INJECTION = "template_injection"
    HEADER_INJECTION = "header_injection"
    PATH_TRAVERSAL = "path_traversal"


@dataclass
class AttackPattern:
    """Attack pattern definition"""
    pattern: Pattern[str]
    attack_type: AttackType
    severity: str  # low, medium, high, critical
    description: str


class SQLInjectionPrevention:
    """Advanced SQL injection prevention system"""
    
    def __init__(self):
        # SQL injection patterns organized by category
        self.sql_patterns = {
            # Union-based injection
            "union_based": [
                re.compile(r'\bunion\b.*\bselect\b', re.IGNORECASE | re.DOTALL),
                re.compile(r'\bunion\b.*\ball\b.*\bselect\b', re.IGNORECASE | re.DOTALL),
                re.compile(r'\bunion\b.*\bdistinct\b.*\bselect\b', re.IGNORECASE | re.DOTALL),
            ],
            
            # Boolean-based blind injection
            "boolean_blind": [
                re.compile(r'\band\b\s*\d+\s*=\s*\d+', re.IGNORECASE),
                re.compile(r'\bor\b\s*\d+\s*=\s*\d+', re.IGNORECASE),
                re.compile(r'\band\b.*\btrue\b', re.IGNORECASE),
                re.compile(r'\bor\b.*\bfalse\b', re.IGNORECASE),
                re.compile(r"'\s*\bor\b\s*'[^']*'\s*=\s*'[^']*'", re.IGNORECASE),
                re.compile(r'"\s*\bor\b\s*"[^"]*"\s*=\s*"[^"]*"', re.IGNORECASE),
            ],
            
            # Time-based blind injection
            "time_blind": [
                re.compile(r'\bwaitfor\b.*\bdelay\b', re.IGNORECASE),
                re.compile(r'\bsleep\b\s*\(\s*\d+\s*\)', re.IGNORECASE),
                re.compile(r'\bbenchmark\b\s*\(\s*\d+', re.IGNORECASE),
                re.compile(r'\bpg_sleep\b\s*\(\s*\d+\s*\)', re.IGNORECASE),
            ],
            
            # Error-based injection
            "error_based": [
                re.compile(r'\bconvert\b.*\bint\b', re.IGNORECASE),
                re.compile(r'\bcast\b.*\bas\b.*\bint\b', re.IGNORECASE),
                re.compile(r'\bextractvalue\b', re.IGNORECASE),
                re.compile(r'\bupdatexml\b', re.IGNORECASE),
            ],
            
            # Stacked queries
            "stacked_queries": [
                re.compile(r';\s*\b(select|insert|update|delete|drop|create|alter|exec|execute)\b', re.IGNORECASE),
                re.compile(r';\s*\bwaitfor\b', re.IGNORECASE),
                re.compile(r';\s*\bshutdown\b', re.IGNORECASE),
            ],
            
            # Out-of-band injection
            "out_of_band": [
                re.compile(r'\bload_file\b', re.IGNORECASE),
                re.compile(r'\binto\b.*\boutfile\b', re.IGNORECASE),
                re.compile(r'\binto\b.*\bdumpfile\b', re.IGNORECASE),
            ],
            
            # Database-specific functions
            "db_functions": [
                re.compile(r'\b(version|user|database|schema)\b\s*\(\s*\)', re.IGNORECASE),
                re.compile(r'\b@@version\b', re.IGNORECASE),
                re.compile(r'\b@@user\b', re.IGNORECASE),
                re.compile(r'\binformation_schema\b', re.IGNORECASE),
                re.compile(r'\bsys\.\w+', re.IGNORECASE),
            ],
            
            # Comment patterns
            "comments": [
                re.compile(r'/\*.*?\*/', re.DOTALL),
                re.compile(r'--\s+.*$', re.MULTILINE),
                re.compile(r'#.*$', re.MULTILINE),
            ],
            
            # Encoding bypass attempts
            "encoding_bypass": [
                re.compile(r'%[0-9a-f]{2}', re.IGNORECASE),
                re.compile(r'\\x[0-9a-f]{2}', re.IGNORECASE),
                re.compile(r'char\s*\(\s*\d+\s*\)', re.IGNORECASE),
                re.compile(r'chr\s*\(\s*\d+\s*\)', re.IGNORECASE),
            ]
        }
        
        # Dangerous SQL keywords that should never appear in user input
        self.dangerous_keywords = {
            'drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update',
            'exec', 'execute', 'sp_', 'xp_', 'cmdshell', 'openrowset', 'bulk',
            'shutdown', 'backup', 'restore', 'grant', 'revoke', 'deny',
            'information_schema', 'sys.', 'master..', 'msdb..', 'tempdb..'
        }
    
    def detect_sql_injection(self, input_text: str) -> List[Dict[str, str]]:
        """Detect SQL injection attempts in input text"""
        if not input_text:
            return []
        
        detections = []
        text_lower = input_text.lower()
        
        # Check each pattern category
        for category, patterns in self.sql_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(input_text)
                if matches:
                    detections.append({
                        "type": "sql_injection",
                        "category": category,
                        "pattern": pattern.pattern,
                        "matches": matches,
                        "severity": self._get_severity(category)
                    })
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword in text_lower:
                detections.append({
                    "type": "sql_injection",
                    "category": "dangerous_keyword",
                    "keyword": keyword,
                    "severity": "high"
                })
        
        return detections
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for SQL injection category"""
        severity_map = {
            "union_based": "critical",
            "boolean_blind": "high",
            "time_blind": "high", 
            "error_based": "high",
            "stacked_queries": "critical",
            "out_of_band": "critical",
            "db_functions": "medium",
            "comments": "medium",
            "encoding_bypass": "high"
        }
        return severity_map.get(category, "medium")
    
    def sanitize_sql_input(self, input_text: str) -> str:
        """Sanitize input to prevent SQL injection"""
        if not input_text:
            return input_text
        
        # Remove SQL comments
        sanitized = re.sub(r'/\*.*?\*/', '', input_text, flags=re.DOTALL)
        sanitized = re.sub(r'--.*$', '', sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r'#.*$', '', sanitized, flags=re.MULTILINE)
        
        # Remove dangerous characters
        sanitized = sanitized.replace(';', '')
        sanitized = sanitized.replace('|', '')
        sanitized = sanitized.replace('&', '')
        
        # Escape single quotes
        sanitized = sanitized.replace("'", "''")
        
        return sanitized.strip()
    
    def validate_table_identifier(self, identifier: str) -> bool:
        """Validate table/column identifier"""
        if not identifier:
            return False
        
        # Check for valid SQL identifier pattern
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', identifier):
            return False
        
        # Check length
        if len(identifier) > 63:  # PostgreSQL limit
            return False
        
        # Check for dangerous keywords
        if identifier.lower() in self.dangerous_keywords:
            return False
        
        return True
    
    def create_safe_query(self, base_query: str, params: Dict[str, Any]) -> str:
        """Create parameterized query safely"""
        # Validate base query structure
        if not self._validate_base_query(base_query):
            raise ValueError("Invalid base query structure")
        
        # Use SQLAlchemy text() with bound parameters
        return str(text(base_query).params(**params))
    
    def _validate_base_query(self, query: str) -> bool:
        """Validate base query structure"""
        # Check for dangerous patterns in the base query
        detections = self.detect_sql_injection(query)
        return len(detections) == 0


class XSSPrevention:
    """Advanced XSS (Cross-Site Scripting) prevention system"""
    
    def __init__(self):
        # XSS attack patterns
        self.xss_patterns = {
            # Script injection
            "script_tags": [
                re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
                re.compile(r'<script[^>]*>', re.IGNORECASE),
                re.compile(r'javascript:', re.IGNORECASE),
                re.compile(r'vbscript:', re.IGNORECASE),
            ],
            
            # Event handlers
            "event_handlers": [
                re.compile(r'on\w+\s*=', re.IGNORECASE),
                re.compile(r'onload\s*=', re.IGNORECASE),
                re.compile(r'onerror\s*=', re.IGNORECASE),
                re.compile(r'onclick\s*=', re.IGNORECASE),
                re.compile(r'onmouseover\s*=', re.IGNORECASE),
                re.compile(r'onfocus\s*=', re.IGNORECASE),
                re.compile(r'onblur\s*=', re.IGNORECASE),
            ],
            
            # Dangerous tags
            "dangerous_tags": [
                re.compile(r'<iframe[^>]*>', re.IGNORECASE),
                re.compile(r'<object[^>]*>', re.IGNORECASE),
                re.compile(r'<embed[^>]*>', re.IGNORECASE),
                re.compile(r'<applet[^>]*>', re.IGNORECASE),
                re.compile(r'<meta[^>]*>', re.IGNORECASE),
                re.compile(r'<link[^>]*>', re.IGNORECASE),
                re.compile(r'<form[^>]*>', re.IGNORECASE),
            ],
            
            # Expression injection
            "expressions": [
                re.compile(r'expression\s*\(', re.IGNORECASE),
                re.compile(r'url\s*\(\s*javascript:', re.IGNORECASE),
                re.compile(r'@import.*javascript:', re.IGNORECASE),
            ],
            
            # Data URIs
            "data_uris": [
                re.compile(r'data:\s*[^;]+;base64', re.IGNORECASE),
                re.compile(r'data:\s*text/html', re.IGNORECASE),
                re.compile(r'data:\s*text/javascript', re.IGNORECASE),
            ],
            
            # Encoded attacks
            "encoded": [
                re.compile(r'&#[x]?[0-9a-f]+;', re.IGNORECASE),
                re.compile(r'%[0-9a-f]{2}', re.IGNORECASE),
                re.compile(r'\\u[0-9a-f]{4}', re.IGNORECASE),
            ]
        }
        
        # Configure bleach cleaner
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'h1', 'h2', 'h3', 
            'h4', 'h5', 'h6', 'blockquote', 'code', 'pre', 'a', 'img'
        ]
        
        self.allowed_attributes = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title', 'width', 'height'],
            '*': ['class', 'id']
        }
        
        self.allowed_protocols = ['http', 'https', 'mailto']
        
        self.cleaner = Cleaner(
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            protocols=self.allowed_protocols,
            strip=True,
            strip_comments=True
        )
    
    def detect_xss(self, input_text: str) -> List[Dict[str, str]]:
        """Detect XSS attempts in input text"""
        if not input_text:
            return []
        
        detections = []
        
        # Check each pattern category
        for category, patterns in self.xss_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(input_text)
                if matches:
                    detections.append({
                        "type": "xss",
                        "category": category,
                        "pattern": pattern.pattern,
                        "matches": matches,
                        "severity": self._get_xss_severity(category)
                    })
        
        return detections
    
    def _get_xss_severity(self, category: str) -> str:
        """Get severity level for XSS category"""
        severity_map = {
            "script_tags": "critical",
            "event_handlers": "high",
            "dangerous_tags": "high",
            "expressions": "high",
            "data_uris": "medium",
            "encoded": "medium"
        }
        return severity_map.get(category, "medium")
    
    def sanitize_html(self, html_content: str, strict: bool = False) -> str:
        """Sanitize HTML content to prevent XSS"""
        if not html_content:
            return html_content
        
        if strict:
            # Strip all HTML tags
            return bleach.clean(html_content, tags=[], attributes={}, strip=True)
        else:
            # Use configured cleaner
            return self.cleaner.clean(html_content)
    
    def escape_html(self, text: str) -> str:
        """HTML-escape user input"""
        if not text:
            return text
        return html.escape(text, quote=True)
    
    def sanitize_javascript(self, js_content: str) -> str:
        """Sanitize JavaScript content (for templates, etc.)"""
        if not js_content:
            return js_content
        
        # Remove dangerous functions and objects
        dangerous_patterns = [
            r'eval\s*\(',
            r'Function\s*\(',
            r'document\.',
            r'window\.',
            r'global\.',
            r'process\.',
            r'require\s*\(',
            r'import\s*\(',
        ]
        
        sanitized = js_content
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '/* BLOCKED */', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def validate_url(self, url: str) -> bool:
        """Validate URL to prevent XSS through redirects"""
        if not url:
            return True
        
        try:
            parsed = urlparse(url)
            
            # Check for dangerous schemes
            dangerous_schemes = ['javascript', 'vbscript', 'data', 'file']
            if parsed.scheme.lower() in dangerous_schemes:
                return False
            
            # Only allow HTTP/HTTPS for external URLs
            if parsed.netloc and parsed.scheme.lower() not in ['http', 'https']:
                return False
            
            return True
            
        except Exception:
            return False


class CSRFProtection:
    """Cross-Site Request Forgery (CSRF) protection system"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_timeout = 3600  # 1 hour
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        timestamp = str(int(datetime.utcnow().timestamp()))
        
        # Create token with timestamp and session ID
        token_data = f"{session_id}:{timestamp}"
        signature = hashlib.hmac_sha256(
            self.secret_key.encode(),
            token_data.encode()
        ).hexdigest()
        
        token = f"{token_data}:{signature}"
        return token
    
    def validate_csrf_token(self, token: str, session_id: str) -> bool:
        """Validate CSRF token"""
        if not token:
            return False
        
        try:
            parts = token.split(':')
            if len(parts) != 3:
                return False
            
            token_session_id, timestamp, signature = parts
            
            # Verify session ID matches
            if token_session_id != session_id:
                return False
            
            # Verify timestamp is not expired
            token_time = int(timestamp)
            current_time = int(datetime.utcnow().timestamp())
            if current_time - token_time > self.token_timeout:
                return False
            
            # Verify signature
            token_data = f"{token_session_id}:{timestamp}"
            expected_signature = hashlib.hmac_sha256(
                self.secret_key.encode(),
                token_data.encode()
            ).hexdigest()
            
            return signature == expected_signature
            
        except (ValueError, TypeError):
            return False


class InjectionPreventionMiddleware(BaseHTTPMiddleware):
    """Comprehensive injection prevention middleware"""
    
    def __init__(self, app, enable_sql_protection: bool = True, enable_xss_protection: bool = True,
                 enable_csrf_protection: bool = True, strict_mode: bool = False):
        super().__init__(app)
        
        self.enable_sql_protection = enable_sql_protection
        self.enable_xss_protection = enable_xss_protection
        self.enable_csrf_protection = enable_csrf_protection
        self.strict_mode = strict_mode
        
        # Initialize protection systems
        self.sql_prevention = SQLInjectionPrevention() if enable_sql_protection else None
        self.xss_prevention = XSSPrevention() if enable_xss_protection else None
        self.csrf_protection = CSRFProtection() if enable_csrf_protection else None
        
        # Excluded paths (for performance)
        self.excluded_paths = [
            "/api/health", "/api/metrics", "/api/docs", "/api/redoc", "/api/openapi.json",
            "/static", "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        """Process request through injection prevention pipeline"""
        
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        try:
            # Validate request for injection attempts
            await self._validate_request(request)
            
            # Process request
            response = await call_next(request)
            
            # Add security headers to response
            self._add_security_headers(response)
            
            return response
            
        except HTTPException as e:
            # Log security violation
            logger.warning(
                f"Security violation detected: {e.detail} "
                f"from {request.client.host} to {request.url.path}"
            )
            raise
        except Exception as e:
            logger.error(f"Injection prevention middleware error: {e}")
            return await call_next(request)
    
    async def _validate_request(self, request: Request):
        """Validate request for injection attempts"""
        
        # Validate query parameters
        for param_name, param_value in request.query_params.items():
            await self._validate_input(param_name, param_value, "query_param")
        
        # Validate path parameters
        for param_name, param_value in request.path_params.items():
            await self._validate_input(param_name, param_value, "path_param")
        
        # Validate headers
        for header_name, header_value in request.headers.items():
            # Skip standard headers
            if header_name.lower() in ['authorization', 'content-type', 'user-agent', 'accept']:
                continue
            await self._validate_input(header_name, header_value, "header")
        
        # Validate request body for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    content_type = request.headers.get("content-type", "")
                    
                    if "application/json" in content_type:
                        try:
                            json_data = json.loads(body.decode('utf-8'))
                            await self._validate_json_data(json_data)
                        except json.JSONDecodeError:
                            pass  # Let the application handle invalid JSON
                    elif "application/x-www-form-urlencoded" in content_type:
                        form_data = await request.form()
                        for field_name, field_value in form_data.items():
                            await self._validate_input(field_name, str(field_value), "form_field")
                    else:
                        # Validate raw body content
                        await self._validate_input("request_body", body.decode('utf-8', errors='ignore'), "body")
            except Exception as e:
                logger.debug(f"Body validation error: {e}")
        
        # CSRF protection for state-changing requests
        if self.csrf_protection and request.method in ["POST", "PUT", "PATCH", "DELETE"]:
            await self._validate_csrf_token(request)
    
    async def _validate_input(self, name: str, value: str, input_type: str):
        """Validate individual input value"""
        if not value:
            return
        
        threats = []
        
        # SQL injection detection
        if self.sql_prevention:
            sql_threats = self.sql_prevention.detect_sql_injection(value)
            threats.extend(sql_threats)
        
        # XSS detection
        if self.xss_prevention:
            xss_threats = self.xss_prevention.detect_xss(value)
            threats.extend(xss_threats)
        
        # Check for critical threats
        critical_threats = [t for t in threats if t.get("severity") == "critical"]
        high_threats = [t for t in threats if t.get("severity") == "high"]
        
        if critical_threats or (self.strict_mode and high_threats):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Security threat detected in {input_type} '{name}'"
            )
        
        # Log medium/low threats
        if threats:
            logger.warning(
                f"Potential security threat in {input_type} '{name}': "
                f"{[t['type'] + ':' + t.get('category', '') for t in threats]}"
            )
    
    async def _validate_json_data(self, data: Union[Dict, List], path: str = ""):
        """Recursively validate JSON data"""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str):
                    await self._validate_input(current_path, value, "json_field")
                elif isinstance(value, (dict, list)):
                    await self._validate_json_data(value, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                if isinstance(item, str):
                    await self._validate_input(current_path, item, "json_array_item")
                elif isinstance(item, (dict, list)):
                    await self._validate_json_data(item, current_path)
    
    async def _validate_csrf_token(self, request: Request):
        """Validate CSRF token for state-changing requests"""
        # Skip CSRF validation for API endpoints with proper authentication
        if request.url.path.startswith("/api/") and request.headers.get("authorization"):
            return
        
        # Get session ID (would typically come from session management system)
        session_id = request.headers.get("x-session-id") or "anonymous"
        
        # Get CSRF token from header or form data
        csrf_token = (
            request.headers.get("x-csrf-token") or
            request.cookies.get("csrf_token")
        )
        
        if not csrf_token or not self.csrf_protection.validate_csrf_token(csrf_token, session_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token validation failed"
            )
    
    def _add_security_headers(self, response: StarletteResponse):
        """Add security headers to response"""
        security_headers = {
            # XSS Protection
            "X-XSS-Protection": "1; mode=block",
            
            # Content Type Options
            "X-Content-Type-Options": "nosniff",
            
            # Frame Options
            "X-Frame-Options": "DENY",
            
            # Content Security Policy (basic)
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value


# Database query safety utilities
class SafeQueryBuilder:
    """Safe SQL query builder with injection prevention"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.sql_prevention = SQLInjectionPrevention()
    
    def safe_select(
        self,
        table_name: str,
        columns: List[str] = None,
        where_conditions: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = None,
        offset: int = None
    ) -> str:
        """Build safe SELECT query with parameterization"""
        
        # Validate table name
        if not self.sql_prevention.validate_table_identifier(table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        
        # Validate column names
        if columns:
            for column in columns:
                if not self.sql_prevention.validate_table_identifier(column):
                    raise ValueError(f"Invalid column name: {column}")
            columns_str = ", ".join(columns)
        else:
            columns_str = "*"
        
        # Build query
        query_parts = [f"SELECT {columns_str} FROM {table_name}"]
        
        # Add WHERE clause
        if where_conditions:
            where_clauses = []
            for column, value in where_conditions.items():
                if not self.sql_prevention.validate_table_identifier(column):
                    raise ValueError(f"Invalid column name in WHERE: {column}")
                where_clauses.append(f"{column} = :{column}")
            
            if where_clauses:
                query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        # Add ORDER BY
        if order_by:
            if not self.sql_prevention.validate_table_identifier(order_by):
                raise ValueError(f"Invalid column name in ORDER BY: {order_by}")
            query_parts.append(f"ORDER BY {order_by}")
        
        # Add LIMIT
        if limit:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError("Invalid LIMIT value")
            query_parts.append(f"LIMIT {limit}")
        
        # Add OFFSET
        if offset:
            if not isinstance(offset, int) or offset < 0:
                raise ValueError("Invalid OFFSET value")
            query_parts.append(f"OFFSET {offset}")
        
        return " ".join(query_parts)
    
    async def execute_safe_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict]:
        """Execute parameterized query safely"""
        try:
            # Use SQLAlchemy text() for parameterized queries
            result = await self.session.execute(text(query), params or {})
            
            # Convert to list of dictionaries
            rows = result.fetchall()
            columns = result.keys()
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Safe query execution error: {e}")
            raise


# Utility functions
def sanitize_html_content(content: str, strict: bool = False) -> str:
    """Utility function to sanitize HTML content"""
    xss_prevention = XSSPrevention()
    return xss_prevention.sanitize_html(content, strict)


def escape_user_input(text: str) -> str:
    """Utility function to escape user input"""
    xss_prevention = XSSPrevention()
    return xss_prevention.escape_html(text)


def validate_sql_identifier(identifier: str) -> bool:
    """Utility function to validate SQL identifier"""
    sql_prevention = SQLInjectionPrevention()
    return sql_prevention.validate_table_identifier(identifier)


def detect_injection_threats(text: str) -> List[Dict[str, str]]:
    """Utility function to detect injection threats"""
    threats = []
    
    sql_prevention = SQLInjectionPrevention()
    threats.extend(sql_prevention.detect_sql_injection(text))
    
    xss_prevention = XSSPrevention()
    threats.extend(xss_prevention.detect_xss(text))
    
    return threats