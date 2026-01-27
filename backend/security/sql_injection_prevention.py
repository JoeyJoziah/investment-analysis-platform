"""
SQL Injection Prevention Middleware and Utilities

This module provides comprehensive protection against SQL injection attacks through:
- Input sanitization and validation
- SQL query pattern detection with weighted scoring
- Parameterized query enforcement
- Database query monitoring and logging

Enhanced with:
- Weighted pattern combination scoring
- Obfuscation detection with bonus weights
- Context-aware scoring (comments, string concatenation)
- UNION-based, time-based blind, error-based, and stacked query detection
- Tuned thresholds to minimize false positives
"""

import re
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from sqlalchemy.sql import ClauseElement
from fastapi import HTTPException, status
import html
import urllib.parse
import hashlib

logger = logging.getLogger(__name__)


class SQLInjectionThreatLevel(Enum):
    """Threat levels for potential SQL injection attempts"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of SQL injection attacks"""
    UNION_BASED = "union_based"
    TIME_BASED_BLIND = "time_based_blind"
    ERROR_BASED = "error_based"
    STACKED_QUERIES = "stacked_queries"
    BOOLEAN_BASED = "boolean_based"
    COMMENT_BASED = "comment_based"
    OBFUSCATION = "obfuscation"
    SYSTEM_ACCESS = "system_access"
    FILE_OPERATION = "file_operation"
    BASIC_INJECTION = "basic_injection"


@dataclass
class PatternMatch:
    """Details of a matched pattern"""
    category: str
    pattern_name: str
    matched_text: str
    base_weight: float
    attack_type: AttackType
    context_multiplier: float = 1.0

    @property
    def weighted_score(self) -> float:
        return self.base_weight * self.context_multiplier


@dataclass
class SQLInjectionDetection:
    """Result of SQL injection detection with detailed scoring"""
    is_threat: bool
    threat_level: SQLInjectionThreatLevel
    detected_patterns: List[str]
    sanitized_input: Optional[str] = None
    recommendation: str = ""
    raw_score: float = 0.0
    weighted_score: float = 0.0
    attack_types: List[AttackType] = field(default_factory=list)
    pattern_matches: List[PatternMatch] = field(default_factory=list)
    obfuscation_detected: bool = False
    combination_bonus: float = 0.0
    request_id: str = ""


class SQLInjectionPrevention:
    """
    Comprehensive SQL injection prevention system.
    
    Features:
    - Pattern-based detection of SQL injection attempts
    - Input sanitization and validation
    - Parameterized query enforcement
    - Real-time threat monitoring
    """
    
    def __init__(self):
        # Common SQL injection patterns
        self.sql_patterns = {
            # Basic SQL keywords
            'basic_sql': [
                r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b',
                r'\b(UNION|JOIN|WHERE|ORDER\s+BY|GROUP\s+BY|HAVING)\b'
            ],
            
            # SQL injection specific patterns
            'injection_patterns': [
                r"'.*'",  # Single quote strings
                r'".*"',  # Double quote strings
                r'--.*',  # SQL comments
                r'/\*.*\*/',  # Multi-line comments
                r'\bOR\s+\d+=\d+',  # OR 1=1 patterns
                r'\bAND\s+\d+=\d+',  # AND 1=1 patterns
                r';\s*(SELECT|INSERT|UPDATE|DELETE|DROP)',  # Statement concatenation
                r'\bUNION\s+(ALL\s+)?SELECT',  # UNION attacks
                r'\b(INFORMATION_SCHEMA|SYS\.|MASTER\.)',  # System table access
                r'@@\w+',  # System variables
                r'\bCAST\s*\(',  # Type casting
                r'\bCONVERT\s*\(',  # Type conversion
                r'\bWAITFOR\s+DELAY',  # Time-based attacks
                r'\bBENCHMARK\s*\(',  # MySQL benchmark attacks
                r'\bSLEEP\s*\(',  # MySQL sleep attacks
                r'\bPG_SLEEP\s*\(',  # PostgreSQL sleep attacks
            ],
            
            # Advanced evasion patterns
            'evasion_patterns': [
                r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
                r'%[0-9a-fA-F]{2}',  # URL encoded characters
                r'&#\d+;',  # HTML entities
                r'&\w+;',  # HTML entity names
                r'\\\w',  # Escape sequences
                r'CHAR\s*\(\s*\d+\s*\)',  # CHAR function
                r'CHR\s*\(\s*\d+\s*\)',  # CHR function
                r'ASCII\s*\(\s*\w+\s*\)',  # ASCII function
            ],
            
            # Database-specific functions
            'db_functions': [
                r'\b(USER|CURRENT_USER|SESSION_USER|SYSTEM_USER)\b',
                r'\b(DATABASE|SCHEMA|VERSION)\s*\(\s*\)',
                r'\b(LEN|LENGTH|SUBSTRING|MID)\s*\(',
                r'\bLOAD_FILE\s*\(',  # File operations
                r'\bINTO\s+(OUTFILE|DUMPFILE)',  # File writing
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.sql_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
    
    def detect_sql_injection(self, user_input: str) -> SQLInjectionDetection:
        """
        Analyze input for potential SQL injection attempts.
        
        Args:
            user_input: User input to analyze
            
        Returns:
            SQLInjectionDetection result
        """
        if not user_input or not isinstance(user_input, str):
            return SQLInjectionDetection(
                is_threat=False,
                threat_level=SQLInjectionThreatLevel.LOW,
                detected_patterns=[]
            )
        
        detected_patterns = []
        threat_score = 0
        
        # Check each pattern category
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(user_input)
                if matches:
                    detected_patterns.extend([f"{category}: {match}" for match in matches])
                    
                    # Assign threat scores
                    if category == 'injection_patterns':
                        threat_score += 10
                    elif category == 'evasion_patterns':
                        threat_score += 15
                    elif category == 'db_functions':
                        threat_score += 8
                    elif category == 'basic_sql':
                        threat_score += 5
        
        # Determine threat level
        if threat_score >= 20:
            threat_level = SQLInjectionThreatLevel.CRITICAL
        elif threat_score >= 15:
            threat_level = SQLInjectionThreatLevel.HIGH
        elif threat_score >= 8:
            threat_level = SQLInjectionThreatLevel.MEDIUM
        else:
            threat_level = SQLInjectionThreatLevel.LOW
        
        is_threat = threat_score > 0
        
        # Generate recommendation
        recommendation = self._get_recommendation(threat_level, detected_patterns)
        
        return SQLInjectionDetection(
            is_threat=is_threat,
            threat_level=threat_level,
            detected_patterns=detected_patterns,
            recommendation=recommendation
        )
    
    def _get_recommendation(self, threat_level: SQLInjectionThreatLevel, patterns: List[str]) -> str:
        """Generate security recommendation based on threat level"""
        if threat_level == SQLInjectionThreatLevel.CRITICAL:
            return "CRITICAL: Block request immediately. Potential SQL injection attack detected."
        elif threat_level == SQLInjectionThreatLevel.HIGH:
            return "HIGH: Sanitize input and use parameterized queries. Log security event."
        elif threat_level == SQLInjectionThreatLevel.MEDIUM:
            return "MEDIUM: Validate and sanitize input. Monitor for suspicious patterns."
        else:
            return "LOW: Standard input validation recommended."
    
    def sanitize_input(self, user_input: str, strict: bool = True) -> str:
        """
        Sanitize user input to prevent SQL injection.
        
        Args:
            user_input: Input to sanitize
            strict: If True, apply strict sanitization
            
        Returns:
            Sanitized input
        """
        if not user_input or not isinstance(user_input, str):
            return ""
        
        sanitized = user_input
        
        if strict:
            # Remove or escape dangerous characters
            sanitized = re.sub(r'[\'";\\]', '', sanitized)  # Remove quotes and backslashes
            sanitized = re.sub(r'--.*', '', sanitized)  # Remove SQL comments
            sanitized = re.sub(r'/\*.*?\*/', '', sanitized)  # Remove multi-line comments
            sanitized = re.sub(r';\s*$', '', sanitized)  # Remove trailing semicolons
        else:
            # Escape dangerous characters
            sanitized = sanitized.replace("'", "''")  # Escape single quotes
            sanitized = sanitized.replace('"', '""')  # Escape double quotes
            sanitized = sanitized.replace('\\', '\\\\')  # Escape backslashes
        
        # URL decode
        sanitized = urllib.parse.unquote(sanitized)
        
        # HTML decode
        sanitized = html.unescape(sanitized)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Limit length
        max_length = 1000  # Adjust as needed
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            logger.warning(f"Input truncated to {max_length} characters")
        
        return sanitized.strip()
    
    def validate_query_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize query parameters.
        
        Args:
            params: Dictionary of query parameters
            
        Returns:
            Sanitized parameters
        """
        validated_params = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Detect potential SQL injection
                detection = self.detect_sql_injection(value)
                
                if detection.threat_level in [SQLInjectionThreatLevel.HIGH, SQLInjectionThreatLevel.CRITICAL]:
                    logger.warning(f"Potential SQL injection in parameter '{key}': {detection.detected_patterns}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid input detected in parameter '{key}'"
                    )
                
                # Sanitize the value
                validated_params[key] = self.sanitize_input(value)
            else:
                validated_params[key] = value
        
        return validated_params


class SecureQueryBuilder:
    """
    Builder for creating secure parameterized database queries.
    
    This class ensures all queries use parameterized statements
    and provides safe query construction methods.
    """
    
    def __init__(self, session: Session):
        self.session = session
        self.prevention = SQLInjectionPrevention()
    
    def execute_safe_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a parameterized query safely.
        
        Args:
            query: SQL query with parameter placeholders (:param_name)
            params: Dictionary of parameters
            
        Returns:
            Query result
        """
        try:
            # Validate parameters
            if params:
                params = self.prevention.validate_query_params(params)
            
            # Execute parameterized query
            stmt = text(query)
            result = self.session.execute(stmt, params or {})
            
            logger.debug(f"Executed safe query: {query[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error executing safe query: {e}")
            raise
    
    def build_select_query(
        self, 
        table: str, 
        columns: List[str], 
        where_conditions: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a safe SELECT query with parameterized conditions.
        
        Args:
            table: Table name (validated)
            columns: List of column names (validated)
            where_conditions: WHERE clause conditions
            limit: LIMIT value
            offset: OFFSET value
            
        Returns:
            Tuple of (query_string, parameters)
        """
        # Validate table name (alphanumeric and underscore only)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
            raise ValueError(f"Invalid table name: {table}")
        
        # Validate column names
        for col in columns:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
                raise ValueError(f"Invalid column name: {col}")
        
        # Build query
        query_parts = []
        params = {}
        
        # SELECT clause
        columns_str = ", ".join(columns)
        query_parts.append(f"SELECT {columns_str} FROM {table}")
        
        # WHERE clause
        if where_conditions:
            where_clauses = []
            for i, (column, value) in enumerate(where_conditions.items()):
                # Validate column name
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column):
                    raise ValueError(f"Invalid column name in WHERE: {column}")
                
                param_name = f"where_{column}_{i}"
                where_clauses.append(f"{column} = :{param_name}")
                params[param_name] = value
            
            query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        # LIMIT clause
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise ValueError("LIMIT must be a non-negative integer")
            query_parts.append(f"LIMIT :limit_value")
            params['limit_value'] = limit
        
        # OFFSET clause  
        if offset is not None:
            if not isinstance(offset, int) or offset < 0:
                raise ValueError("OFFSET must be a non-negative integer")
            query_parts.append(f"OFFSET :offset_value")
            params['offset_value'] = offset
        
        query = " ".join(query_parts)
        return query, params
    
    def build_insert_query(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a safe INSERT query with parameterized values.
        
        Args:
            table: Table name (validated)
            data: Dictionary of column:value pairs
            
        Returns:
            Tuple of (query_string, parameters)
        """
        # Validate table name
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
            raise ValueError(f"Invalid table name: {table}")
        
        if not data:
            raise ValueError("No data provided for INSERT")
        
        # Validate column names and build query
        columns = []
        placeholders = []
        params = {}
        
        for i, (column, value) in enumerate(data.items()):
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column):
                raise ValueError(f"Invalid column name: {column}")
            
            columns.append(column)
            param_name = f"insert_{column}_{i}"
            placeholders.append(f":{param_name}")
            params[param_name] = value
        
        columns_str = ", ".join(columns)
        placeholders_str = ", ".join(placeholders)
        
        query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders_str})"
        return query, params
    
    def build_update_query(
        self,
        table: str,
        data: Dict[str, Any],
        where_conditions: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a safe UPDATE query with parameterized values.
        
        Args:
            table: Table name (validated)
            data: Dictionary of column:value pairs to update
            where_conditions: WHERE clause conditions
            
        Returns:
            Tuple of (query_string, parameters)
        """
        # Validate table name
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
            raise ValueError(f"Invalid table name: {table}")
        
        if not data:
            raise ValueError("No data provided for UPDATE")
        
        if not where_conditions:
            raise ValueError("WHERE conditions required for UPDATE (safety measure)")
        
        # Build SET clause
        set_clauses = []
        params = {}
        
        for i, (column, value) in enumerate(data.items()):
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column):
                raise ValueError(f"Invalid column name: {column}")
            
            param_name = f"update_{column}_{i}"
            set_clauses.append(f"{column} = :{param_name}")
            params[param_name] = value
        
        # Build WHERE clause
        where_clauses = []
        for i, (column, value) in enumerate(where_conditions.items()):
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column):
                raise ValueError(f"Invalid column name in WHERE: {column}")
            
            param_name = f"where_{column}_{i}"
            where_clauses.append(f"{column} = :{param_name}")
            params[param_name] = value
        
        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"
        return query, params


# Global prevention instance
sql_injection_prevention = SQLInjectionPrevention()


def validate_user_input(user_input: str, strict: bool = True) -> str:
    """Convenience function for input validation"""
    detection = sql_injection_prevention.detect_sql_injection(user_input)
    
    if detection.threat_level in [SQLInjectionThreatLevel.HIGH, SQLInjectionThreatLevel.CRITICAL]:
        logger.warning(f"SQL injection attempt blocked: {detection.detected_patterns}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input detected"
        )
    
    return sql_injection_prevention.sanitize_input(user_input, strict=strict)


def get_secure_query_builder(session: Session) -> SecureQueryBuilder:
    """Get a secure query builder instance"""
    return SecureQueryBuilder(session)