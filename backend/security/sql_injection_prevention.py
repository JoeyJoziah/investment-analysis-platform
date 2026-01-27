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
    Comprehensive SQL injection prevention system with weighted scoring.

    Features:
    - Pattern-based detection with weighted combination scoring
    - Attack type classification (UNION, time-based, error-based, stacked)
    - Obfuscation detection with bonus weights
    - Context-aware scoring (comments, string concatenation)
    - Tuned thresholds for minimal false positives
    - Detailed logging of detected attempts
    """

    # Thresholds tuned to minimize false positives
    THRESHOLD_CRITICAL = 50.0  # High confidence attack
    THRESHOLD_HIGH = 30.0      # Likely attack
    THRESHOLD_MEDIUM = 15.0    # Suspicious, needs review
    THRESHOLD_LOW = 5.0        # Minor concern

    # Combination bonuses when multiple attack types detected
    COMBINATION_BONUS_TWO_TYPES = 10.0
    COMBINATION_BONUS_THREE_PLUS = 25.0

    # Obfuscation multiplier
    OBFUSCATION_MULTIPLIER = 1.5

    def __init__(self):
        # Pattern definitions with weights and attack types
        # Format: (pattern, base_weight, attack_type, pattern_name)
        self._pattern_definitions = {
            'union_based': [
                (r'\bUNION\s+(ALL\s+)?SELECT\b', 20.0, AttackType.UNION_BASED, 'union_select'),
                (r'\bUNION\s+SELECT\s+NULL', 25.0, AttackType.UNION_BASED, 'union_null_probe'),
                (r'\bORDER\s+BY\s+\d+\s*--', 15.0, AttackType.UNION_BASED, 'order_by_probe'),
                (r'\bUNION\s+(ALL\s+)?SELECT\s+\d+,\s*\d+', 22.0, AttackType.UNION_BASED, 'union_column_enum'),
                (r"'\s*UNION\s+SELECT", 25.0, AttackType.UNION_BASED, 'quote_union'),
            ],

            'time_based_blind': [
                (r'\bWAITFOR\s+DELAY\s+[\'"]?\d+:\d+:\d+', 30.0, AttackType.TIME_BASED_BLIND, 'waitfor_delay'),
                (r'\bSLEEP\s*\(\s*\d+\s*\)', 30.0, AttackType.TIME_BASED_BLIND, 'mysql_sleep'),
                (r'\bPG_SLEEP\s*\(\s*\d+\s*\)', 30.0, AttackType.TIME_BASED_BLIND, 'pg_sleep'),
                (r'\bBENCHMARK\s*\(\s*\d+', 28.0, AttackType.TIME_BASED_BLIND, 'mysql_benchmark'),
                (r'\bDBMS_LOCK\.SLEEP', 30.0, AttackType.TIME_BASED_BLIND, 'oracle_sleep'),
                (r"IF\s*\(.*\)\s*WAITFOR", 32.0, AttackType.TIME_BASED_BLIND, 'conditional_waitfor'),
                (r"CASE\s+WHEN.*THEN\s+SLEEP", 32.0, AttackType.TIME_BASED_BLIND, 'case_sleep'),
            ],

            'error_based': [
                (r'\bEXTRACTVALUE\s*\(', 25.0, AttackType.ERROR_BASED, 'extractvalue'),
                (r'\bUPDATEXML\s*\(', 25.0, AttackType.ERROR_BASED, 'updatexml'),
                (r'\bEXP\s*\(\s*~\s*\(', 25.0, AttackType.ERROR_BASED, 'exp_error'),
                (r'\bCONVERT\s*\(\s*INT', 18.0, AttackType.ERROR_BASED, 'convert_error'),
                (r'\bCAST\s*\([^)]+\s+AS\s+INT\)', 15.0, AttackType.ERROR_BASED, 'cast_error'),
                (r"'[^']*\+\s*@@version", 22.0, AttackType.ERROR_BASED, 'version_concat'),
                (r'\bGROUP\s+BY\s+.+\s+HAVING', 12.0, AttackType.ERROR_BASED, 'having_probe'),
                (r'\bRAISERROR\s*\(', 20.0, AttackType.ERROR_BASED, 'raiserror'),
            ],

            'stacked_queries': [
                (r';\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|TRUNCATE)\b', 28.0, AttackType.STACKED_QUERIES, 'stacked_dml'),
                (r';\s*DECLARE\s+@', 30.0, AttackType.STACKED_QUERIES, 'declare_var'),
                (r';\s*SET\s+@', 25.0, AttackType.STACKED_QUERIES, 'set_var'),
                (r';\s*EXEC\s*\(', 30.0, AttackType.STACKED_QUERIES, 'exec_stacked'),
                (r';\s*xp_cmdshell', 40.0, AttackType.STACKED_QUERIES, 'xp_cmdshell'),
                (r';\s*sp_executesql', 30.0, AttackType.STACKED_QUERIES, 'sp_executesql'),
                (r"'\s*;\s*DROP\s+TABLE", 35.0, AttackType.STACKED_QUERIES, 'drop_table'),
                (r"'\s*;\s*SHUTDOWN", 40.0, AttackType.STACKED_QUERIES, 'shutdown'),
            ],

            'boolean_based': [
                (r"\bOR\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?", 15.0, AttackType.BOOLEAN_BASED, 'or_equals'),
                (r"\bAND\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?", 12.0, AttackType.BOOLEAN_BASED, 'and_equals'),
                (r"\bOR\s+['\"]?[a-z]+['\"]?\s*=\s*['\"]?[a-z]+['\"]?", 18.0, AttackType.BOOLEAN_BASED, 'or_string_equals'),
                (r"'\s*OR\s*'1'\s*=\s*'1", 20.0, AttackType.BOOLEAN_BASED, 'classic_or_true'),
                (r"'\s*OR\s*1\s*=\s*1\s*--", 22.0, AttackType.BOOLEAN_BASED, 'or_true_comment'),
                (r"'\s*AND\s*'1'\s*=\s*'2", 15.0, AttackType.BOOLEAN_BASED, 'and_false'),
                (r"\bOR\s+NOT\s+\d+", 12.0, AttackType.BOOLEAN_BASED, 'or_not'),
                (r"LIKE\s+['\"]%", 3.0, AttackType.BOOLEAN_BASED, 'like_wildcard'),  # Low weight, common legitimate use
            ],

            'comment_based': [
                (r'--\s*$', 8.0, AttackType.COMMENT_BASED, 'line_comment_end'),
                (r"--\s*['\"]", 12.0, AttackType.COMMENT_BASED, 'comment_after_quote'),
                (r'/\*.*?\*/', 6.0, AttackType.COMMENT_BASED, 'block_comment'),
                (r'/\*![0-9]*', 15.0, AttackType.COMMENT_BASED, 'mysql_conditional'),
                (r'#\s*$', 8.0, AttackType.COMMENT_BASED, 'hash_comment'),
                (r"'\s*/\*", 12.0, AttackType.COMMENT_BASED, 'quote_block_comment'),
            ],

            'obfuscation': [
                (r'CHAR\s*\(\s*\d+\s*\)', 12.0, AttackType.OBFUSCATION, 'char_function'),
                (r'CHR\s*\(\s*\d+\s*\)', 12.0, AttackType.OBFUSCATION, 'chr_function'),
                (r'CONCAT\s*\([^)]*CHAR\s*\(', 18.0, AttackType.OBFUSCATION, 'concat_char'),
                (r'0x[0-9a-fA-F]{2,}', 10.0, AttackType.OBFUSCATION, 'hex_encoding'),
                (r'%[0-9a-fA-F]{2}', 5.0, AttackType.OBFUSCATION, 'url_encoding'),
                (r'&#x?[0-9a-fA-F]+;', 8.0, AttackType.OBFUSCATION, 'html_entity'),
                (r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', 15.0, AttackType.OBFUSCATION, 'control_chars'),
                (r'\bASCII\s*\(', 8.0, AttackType.OBFUSCATION, 'ascii_function'),
                (r'(?i)s\s*e\s*l\s*e\s*c\s*t', 20.0, AttackType.OBFUSCATION, 'spaced_keyword'),
                (r'(?i)u\s*n\s*i\s*o\s*n', 20.0, AttackType.OBFUSCATION, 'spaced_union'),
                (r'\+\s*[\'"]', 8.0, AttackType.OBFUSCATION, 'string_concat'),
            ],

            'system_access': [
                (r'\b(INFORMATION_SCHEMA|SYS\.|MASTER\.)', 20.0, AttackType.SYSTEM_ACCESS, 'system_tables'),
                (r'@@(VERSION|SERVERNAME|LANGUAGE)', 15.0, AttackType.SYSTEM_ACCESS, 'system_vars'),
                (r'\b(USER|CURRENT_USER|SESSION_USER|SYSTEM_USER)\s*\(\s*\)', 12.0, AttackType.SYSTEM_ACCESS, 'user_functions'),
                (r'\b(DATABASE|SCHEMA|VERSION)\s*\(\s*\)', 10.0, AttackType.SYSTEM_ACCESS, 'db_functions'),
                (r'\bTABLE_NAME\b', 15.0, AttackType.SYSTEM_ACCESS, 'table_enumeration'),
                (r'\bCOLUMN_NAME\b', 15.0, AttackType.SYSTEM_ACCESS, 'column_enumeration'),
                (r'\bSELECT\s+.*\s+FROM\s+.*\.tables', 22.0, AttackType.SYSTEM_ACCESS, 'table_query'),
            ],

            'file_operation': [
                (r'\bLOAD_FILE\s*\(', 30.0, AttackType.FILE_OPERATION, 'load_file'),
                (r'\bINTO\s+(OUTFILE|DUMPFILE)', 35.0, AttackType.FILE_OPERATION, 'file_write'),
                (r'\bUTL_FILE\b', 30.0, AttackType.FILE_OPERATION, 'oracle_file'),
                (r'\bBFILE\s*\(', 30.0, AttackType.FILE_OPERATION, 'bfile'),
            ],

            'basic_injection': [
                (r"'\s*$", 3.0, AttackType.BASIC_INJECTION, 'trailing_quote'),
                (r'"\s*$', 3.0, AttackType.BASIC_INJECTION, 'trailing_dquote'),
                (r"''\s*$", 2.0, AttackType.BASIC_INJECTION, 'escaped_quote'),  # Often legitimate
                (r'\bSELECT\s+\*\s+FROM\b', 8.0, AttackType.BASIC_INJECTION, 'select_star'),
                (r'\bDROP\s+(TABLE|DATABASE)\b', 30.0, AttackType.BASIC_INJECTION, 'drop_statement'),
                (r'\bTRUNCATE\s+TABLE\b', 28.0, AttackType.BASIC_INJECTION, 'truncate'),
                (r'\bDELETE\s+FROM\b', 15.0, AttackType.BASIC_INJECTION, 'delete_from'),
                (r'\bINSERT\s+INTO\b', 8.0, AttackType.BASIC_INJECTION, 'insert_into'),
            ],
        }

        # Compile patterns for efficiency
        self.compiled_patterns: Dict[str, List[Tuple[re.Pattern, float, AttackType, str]]] = {}
        for category, patterns in self._pattern_definitions.items():
            self.compiled_patterns[category] = [
                (re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL), weight, attack_type, name)
                for pattern, weight, attack_type, name in patterns
            ]

        # Whitelist patterns for legitimate use (reduce false positives)
        self._whitelist_patterns = [
            re.compile(r'^[a-zA-Z0-9_\-\s@.]+$'),  # Simple alphanumeric with common chars
            re.compile(r'^[a-zA-Z0-9\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z]+$'),  # Email addresses
            re.compile(r'^\d{4}-\d{2}-\d{2}$'),  # Date format
            re.compile(r'^[a-zA-Z0-9\-]{36}$'),  # UUIDs
        ]

        # Context indicators that increase suspicion
        self._context_indicators = {
            'has_quote_before_keyword': (r"['\"].*\b(SELECT|UNION|OR|AND)\b", 1.3),
            'has_comment_terminator': (r"(--|#|/\*)\s*$", 1.4),
            'multiple_semicolons': (r';.*?;', 1.5),
            'nested_functions': (r'\([^)]*\([^)]*\)', 1.2),
            'long_hex_string': (r'0x[0-9a-fA-F]{20,}', 1.6),
        }

        # Compile context indicators
        self._compiled_context = {
            name: (re.compile(pattern, re.IGNORECASE), multiplier)
            for name, (pattern, multiplier) in self._context_indicators.items()
        }
    
    def _generate_request_id(self, user_input: str) -> str:
        """Generate a unique request ID for tracking"""
        timestamp = str(time.time())
        hash_input = f"{timestamp}-{user_input[:50]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _is_whitelisted(self, user_input: str) -> bool:
        """Check if input matches whitelist patterns (reduces false positives)"""
        for pattern in self._whitelist_patterns:
            if pattern.fullmatch(user_input):
                return True
        return False

    def _calculate_context_multiplier(self, user_input: str) -> Tuple[float, List[str]]:
        """
        Calculate context-based multiplier for weighted scoring.
        Returns multiplier and list of context indicators found.
        """
        multiplier = 1.0
        indicators_found = []

        for name, (pattern, mult) in self._compiled_context.items():
            if pattern.search(user_input):
                multiplier *= mult
                indicators_found.append(name)

        return multiplier, indicators_found

    def _detect_obfuscation(self, user_input: str) -> Tuple[bool, float]:
        """
        Detect obfuscation attempts and return bonus weight.
        Returns (obfuscation_detected, bonus_multiplier)
        """
        obfuscation_count = 0

        for pattern, weight, attack_type, name in self.compiled_patterns.get('obfuscation', []):
            if pattern.search(user_input):
                obfuscation_count += 1

        if obfuscation_count >= 3:
            return True, self.OBFUSCATION_MULTIPLIER
        elif obfuscation_count >= 1:
            return True, 1.0 + (obfuscation_count * 0.15)
        return False, 1.0

    def _calculate_combination_bonus(self, attack_types: Set[AttackType]) -> float:
        """
        Calculate bonus for multiple attack types detected.
        Multiple attack vectors indicate sophisticated attacks.
        """
        # Exclude basic types from combination bonus
        significant_types = attack_types - {AttackType.BASIC_INJECTION, AttackType.COMMENT_BASED}

        if len(significant_types) >= 3:
            return self.COMBINATION_BONUS_THREE_PLUS
        elif len(significant_types) >= 2:
            return self.COMBINATION_BONUS_TWO_TYPES
        return 0.0

    def detect_sql_injection(self, user_input: str) -> SQLInjectionDetection:
        """
        Analyze input for potential SQL injection attempts using weighted scoring.

        The scoring system uses:
        - Base weights for each pattern (severity-based)
        - Context multipliers (comments, string concatenation)
        - Obfuscation bonus (attempts to hide attack)
        - Combination bonus (multiple attack types)

        Args:
            user_input: User input to analyze

        Returns:
            SQLInjectionDetection with detailed scoring information
        """
        request_id = self._generate_request_id(user_input or "")

        if not user_input or not isinstance(user_input, str):
            return SQLInjectionDetection(
                is_threat=False,
                threat_level=SQLInjectionThreatLevel.NONE,
                detected_patterns=[],
                request_id=request_id
            )

        # Quick whitelist check to reduce false positives
        if self._is_whitelisted(user_input):
            return SQLInjectionDetection(
                is_threat=False,
                threat_level=SQLInjectionThreatLevel.NONE,
                detected_patterns=[],
                request_id=request_id
            )

        pattern_matches: List[PatternMatch] = []
        detected_patterns: List[str] = []
        attack_types_found: Set[AttackType] = set()
        raw_score = 0.0

        # Calculate context multiplier
        context_multiplier, context_indicators = self._calculate_context_multiplier(user_input)

        # Check each pattern category
        for category, patterns in self.compiled_patterns.items():
            for pattern, base_weight, attack_type, pattern_name in patterns:
                matches = pattern.findall(user_input)
                if matches:
                    for match in matches:
                        match_text = match if isinstance(match, str) else str(match)
                        pattern_match = PatternMatch(
                            category=category,
                            pattern_name=pattern_name,
                            matched_text=match_text[:100],  # Truncate for logging
                            base_weight=base_weight,
                            attack_type=attack_type,
                            context_multiplier=context_multiplier
                        )
                        pattern_matches.append(pattern_match)
                        detected_patterns.append(f"{category}:{pattern_name}:{match_text[:50]}")
                        attack_types_found.add(attack_type)
                        raw_score += base_weight

        # Detect obfuscation and apply multiplier
        obfuscation_detected, obfuscation_multiplier = self._detect_obfuscation(user_input)

        # Calculate combination bonus
        combination_bonus = self._calculate_combination_bonus(attack_types_found)

        # Calculate final weighted score
        weighted_score = (raw_score * context_multiplier * obfuscation_multiplier) + combination_bonus

        # Determine threat level based on weighted score
        threat_level = self._determine_threat_level(weighted_score, attack_types_found)

        is_threat = threat_level not in [SQLInjectionThreatLevel.NONE, SQLInjectionThreatLevel.LOW]

        # Generate recommendation
        recommendation = self._get_recommendation(threat_level, attack_types_found, weighted_score)

        # Log detected attempts
        if is_threat:
            self._log_detection(
                request_id=request_id,
                user_input=user_input,
                threat_level=threat_level,
                weighted_score=weighted_score,
                attack_types=attack_types_found,
                pattern_matches=pattern_matches,
                context_indicators=context_indicators,
                obfuscation_detected=obfuscation_detected
            )

        return SQLInjectionDetection(
            is_threat=is_threat,
            threat_level=threat_level,
            detected_patterns=detected_patterns,
            recommendation=recommendation,
            raw_score=raw_score,
            weighted_score=weighted_score,
            attack_types=list(attack_types_found),
            pattern_matches=pattern_matches,
            obfuscation_detected=obfuscation_detected,
            combination_bonus=combination_bonus,
            request_id=request_id
        )

    def _determine_threat_level(
        self,
        weighted_score: float,
        attack_types: Set[AttackType]
    ) -> SQLInjectionThreatLevel:
        """
        Determine threat level based on weighted score and attack types.
        Critical attacks (stacked queries, file operations) are escalated.
        """
        # Immediate escalation for critical attack types regardless of score
        critical_types = {AttackType.FILE_OPERATION, AttackType.STACKED_QUERIES}
        if attack_types & critical_types and weighted_score >= self.THRESHOLD_MEDIUM:
            return SQLInjectionThreatLevel.CRITICAL

        # Score-based threat level
        if weighted_score >= self.THRESHOLD_CRITICAL:
            return SQLInjectionThreatLevel.CRITICAL
        elif weighted_score >= self.THRESHOLD_HIGH:
            return SQLInjectionThreatLevel.HIGH
        elif weighted_score >= self.THRESHOLD_MEDIUM:
            return SQLInjectionThreatLevel.MEDIUM
        elif weighted_score >= self.THRESHOLD_LOW:
            return SQLInjectionThreatLevel.LOW
        else:
            return SQLInjectionThreatLevel.NONE

    def _get_recommendation(
        self,
        threat_level: SQLInjectionThreatLevel,
        attack_types: Set[AttackType],
        score: float
    ) -> str:
        """Generate detailed security recommendation based on threat analysis"""
        attack_type_names = [at.value for at in attack_types] if attack_types else []

        if threat_level == SQLInjectionThreatLevel.CRITICAL:
            return (
                f"CRITICAL: Block request immediately. SQL injection attack detected "
                f"(score: {score:.1f}). Attack types: {', '.join(attack_type_names)}. "
                "Log incident and consider IP blocking."
            )
        elif threat_level == SQLInjectionThreatLevel.HIGH:
            return (
                f"HIGH: Sanitize input and use parameterized queries (score: {score:.1f}). "
                f"Detected patterns: {', '.join(attack_type_names)}. Log security event."
            )
        elif threat_level == SQLInjectionThreatLevel.MEDIUM:
            return (
                f"MEDIUM: Validate and sanitize input (score: {score:.1f}). "
                "Monitor for repeated suspicious patterns from this source."
            )
        elif threat_level == SQLInjectionThreatLevel.LOW:
            return f"LOW: Standard input validation recommended (score: {score:.1f})."
        else:
            return "NONE: No threats detected."

    def _log_detection(
        self,
        request_id: str,
        user_input: str,
        threat_level: SQLInjectionThreatLevel,
        weighted_score: float,
        attack_types: Set[AttackType],
        pattern_matches: List[PatternMatch],
        context_indicators: List[str],
        obfuscation_detected: bool
    ) -> None:
        """
        Log detected SQL injection attempts with detailed information.
        """
        # Truncate input for logging (avoid log injection)
        safe_input = user_input[:200].replace('\n', '\\n').replace('\r', '\\r')
        attack_type_names = [at.value for at in attack_types]

        log_data = {
            "request_id": request_id,
            "threat_level": threat_level.value,
            "weighted_score": round(weighted_score, 2),
            "attack_types": attack_type_names,
            "pattern_count": len(pattern_matches),
            "context_indicators": context_indicators,
            "obfuscation_detected": obfuscation_detected,
            "input_preview": safe_input,
        }

        if threat_level == SQLInjectionThreatLevel.CRITICAL:
            logger.critical(
                f"SQL INJECTION ATTACK DETECTED [CRITICAL] - "
                f"request_id={request_id} score={weighted_score:.1f} "
                f"types={attack_type_names} input='{safe_input}'"
            )
            # Also log structured data for security monitoring
            logger.critical(f"SQL_INJECTION_DETAIL: {log_data}")

        elif threat_level == SQLInjectionThreatLevel.HIGH:
            logger.warning(
                f"SQL injection attempt [HIGH] - "
                f"request_id={request_id} score={weighted_score:.1f} "
                f"types={attack_type_names}"
            )
            logger.warning(f"SQL_INJECTION_DETAIL: {log_data}")

        elif threat_level == SQLInjectionThreatLevel.MEDIUM:
            logger.info(
                f"Suspicious SQL pattern [MEDIUM] - "
                f"request_id={request_id} score={weighted_score:.1f}"
            )

        # Log individual high-weight patterns for forensics
        for match in pattern_matches:
            if match.base_weight >= 20.0:
                logger.warning(
                    f"High-weight pattern detected: {match.pattern_name} "
                    f"(weight={match.base_weight}) in request {request_id}"
                )
    
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
        Validate and sanitize query parameters with enhanced logging.

        Args:
            params: Dictionary of query parameters

        Returns:
            Sanitized parameters

        Raises:
            HTTPException: If HIGH or CRITICAL threat level detected
        """
        validated_params = {}

        for key, value in params.items():
            if isinstance(value, str):
                # Detect potential SQL injection
                detection = self.detect_sql_injection(value)

                if detection.threat_level in [SQLInjectionThreatLevel.HIGH, SQLInjectionThreatLevel.CRITICAL]:
                    logger.warning(
                        f"SQL injection blocked in parameter '{key}' - "
                        f"request_id={detection.request_id} "
                        f"threat_level={detection.threat_level.value} "
                        f"score={detection.weighted_score:.1f} "
                        f"attack_types={[at.value for at in detection.attack_types]}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid input detected in parameter '{key}'"
                    )

                # Log medium threats for monitoring
                if detection.threat_level == SQLInjectionThreatLevel.MEDIUM:
                    logger.info(
                        f"Suspicious pattern in parameter '{key}' - "
                        f"request_id={detection.request_id} "
                        f"score={detection.weighted_score:.1f}"
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
    """
    Convenience function for input validation with enhanced SQL injection detection.

    Args:
        user_input: User-provided input string to validate
        strict: If True, apply strict sanitization (removes dangerous chars)

    Returns:
        Sanitized input string

    Raises:
        HTTPException: If HIGH or CRITICAL threat level detected
    """
    detection = sql_injection_prevention.detect_sql_injection(user_input)

    if detection.threat_level in [SQLInjectionThreatLevel.HIGH, SQLInjectionThreatLevel.CRITICAL]:
        logger.warning(
            f"SQL injection attempt blocked - "
            f"request_id={detection.request_id} "
            f"threat_level={detection.threat_level.value} "
            f"score={detection.weighted_score:.1f} "
            f"attack_types={[at.value for at in detection.attack_types]} "
            f"obfuscation={detection.obfuscation_detected}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input detected"
        )

    return sql_injection_prevention.sanitize_input(user_input, strict=strict)


def get_detection_result(user_input: str) -> SQLInjectionDetection:
    """
    Get detailed SQL injection detection result without blocking.

    Useful for logging, monitoring, or custom handling of detection results.

    Args:
        user_input: User-provided input string to analyze

    Returns:
        SQLInjectionDetection with full scoring details
    """
    return sql_injection_prevention.detect_sql_injection(user_input)


def get_secure_query_builder(session: Session) -> SecureQueryBuilder:
    """Get a secure query builder instance"""
    return SecureQueryBuilder(session)