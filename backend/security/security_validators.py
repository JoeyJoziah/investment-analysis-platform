"""
Security validators extracted from security_config (#99 / Wave 12).

Password, API key, file-upload, and scanner helpers that depend on
SecurityConfig policy constants.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import secrets
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _security_config():
    """Lazy import to avoid circular dependency with security_config re-exports."""
    from backend.security.security_config import SecurityConfig
    return SecurityConfig


class _SC:
    """Attribute proxy for SecurityConfig constants."""

    def __getattr__(self, name: str):
        return getattr(_security_config(), name)


SecurityConfig = _SC()  # type: ignore[misc, assignment]


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
        """Generate a cryptographically secure password"""
        import string

        # Ensure we have all required character types
        chars = ""
        password = ""

        if SecurityConfig.PASSWORD_REQUIRE_UPPERCASE:
            chars += string.ascii_uppercase
            password += secrets.choice(string.ascii_uppercase)

        if SecurityConfig.PASSWORD_REQUIRE_LOWERCASE:
            chars += string.ascii_lowercase
            password += secrets.choice(string.ascii_lowercase)

        if SecurityConfig.PASSWORD_REQUIRE_DIGITS:
            chars += string.digits
            password += secrets.choice(string.digits)

        if SecurityConfig.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            chars += special_chars
            password += secrets.choice(special_chars)

        # Fill remaining length with cryptographically secure random choices
        for _ in range(length - len(password)):
            password += secrets.choice(chars)

        # Shuffle the password using Fisher-Yates with secure randomness
        password_list = list(password)
        for i in range(len(password_list) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            password_list[i], password_list[j] = password_list[j], password_list[i]

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


class FileUploadValidator:
    """
    Comprehensive file upload validation with MIME type detection.
    Validates that actual file content matches claimed extension to prevent
    attackers from bypassing security by renaming malicious files.
    """

    # Logger for security monitoring
    _logger = logging.getLogger("security.file_upload")

    @classmethod
    def detect_mime_type_from_content(cls, file_content: bytes) -> Optional[str]:
        """
        Detect MIME type from file content using magic bytes (file signatures).
        Returns None if the content doesn't match any known signature.
        """
        for mime_type, signatures in SecurityConfig.FILE_SIGNATURES.items():
            for signature in signatures:
                if file_content.startswith(signature):
                    return mime_type
        return None

    @classmethod
    def detect_mime_type_from_extension(cls, filename: str) -> Optional[str]:
        """
        Detect MIME type from file extension using mimetypes library.
        Returns None if extension is unknown.
        """
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type

    @classmethod
    def is_text_based_file(cls, extension: str) -> bool:
        """Check if file type is text-based (CSV, JSON, TXT) which may not have magic bytes."""
        return extension.lower() in [".csv", ".json", ".txt"]

    @classmethod
    def validate_text_content(cls, file_content: bytes, extension: str) -> Tuple[bool, Optional[str]]:
        """
        Validate text-based files by checking content structure.
        Returns (is_valid, error_message).
        """
        try:
            # Attempt to decode as UTF-8
            text_content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Try latin-1 as fallback
                text_content = file_content.decode("latin-1")
            except Exception:
                return False, "File content is not valid text"

        ext_lower = extension.lower()

        if ext_lower == ".json":
            import json
            try:
                json.loads(text_content)
                return True, None
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON structure: {str(e)[:100]}"

        if ext_lower == ".csv":
            # Basic CSV validation - check for reasonable structure
            lines = text_content.strip().split("\n")
            if len(lines) == 0:
                return False, "Empty CSV file"
            # Check that it has some comma or tab delimiters (common CSV patterns)
            first_line = lines[0]
            if "," not in first_line and "\t" not in first_line and ";" not in first_line:
                # Single column CSV is technically valid
                pass
            return True, None

        if ext_lower == ".txt":
            # Plain text - just ensure it's decodable (already done above)
            return True, None

        return True, None

    @classmethod
    def validate_mime_type(
        cls,
        file_content: bytes,
        filename: str,
        claimed_content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate that file content matches its claimed type.

        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename with extension
            claimed_content_type: Content-Type header from upload (optional)

        Returns:
            Dict with validation results:
                - valid: bool - whether the file passed validation
                - detected_mime: str - MIME type detected from content
                - expected_mime: str - MIME type expected from extension
                - issues: List[str] - list of validation issues
                - extension: str - file extension
        """
        result = {
            "valid": True,
            "detected_mime": None,
            "expected_mime": None,
            "claimed_mime": claimed_content_type,
            "issues": [],
            "extension": None,
            "filename": filename
        }

        # Extract and validate extension
        _, ext = os.path.splitext(filename.lower())
        result["extension"] = ext

        if not ext:
            result["valid"] = False
            result["issues"].append("File has no extension")
            cls._log_rejected_upload(filename, result["issues"], "no_extension")
            return result

        # Check if extension is in allowlist
        if ext not in SecurityConfig.ALLOWED_FILE_TYPES:
            result["valid"] = False
            result["issues"].append(f"File extension '{ext}' is not in allowed list")
            cls._log_rejected_upload(filename, result["issues"], "disallowed_extension")
            return result

        # Get expected MIME types for this extension
        expected_mimes = SecurityConfig.ALLOWED_MIME_TYPES.get(ext, [])
        result["expected_mime"] = expected_mimes[0] if expected_mimes else None

        # Detect MIME type from content
        detected_mime = cls.detect_mime_type_from_content(file_content)
        result["detected_mime"] = detected_mime

        # Handle text-based files specially (they don't have magic bytes)
        if cls.is_text_based_file(ext):
            is_valid_text, text_error = cls.validate_text_content(file_content, ext)
            if not is_valid_text:
                result["valid"] = False
                result["issues"].append(text_error)
                cls._log_rejected_upload(filename, result["issues"], "invalid_text_content")
                return result

            # For text files, also check they don't contain binary/executable signatures
            if detected_mime and detected_mime not in ["text/plain", "text/csv", "application/json"]:
                result["valid"] = False
                result["issues"].append(
                    f"File claims to be {ext} but contains binary content signature for '{detected_mime}'"
                )
                cls._log_rejected_upload(filename, result["issues"], "mime_mismatch_binary_in_text")
                return result

            # Text file validation passed
            result["detected_mime"] = expected_mimes[0] if expected_mimes else "text/plain"
            return result

        # For binary files, detected MIME must be present
        if detected_mime is None:
            # Check if it might be a ZIP-based format (xlsx, docx, etc.)
            if file_content.startswith(b"PK\x03\x04"):
                if ext == ".xlsx":
                    result["detected_mime"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:
                    result["valid"] = False
                    result["issues"].append(
                        f"File appears to be a ZIP archive but extension is '{ext}'"
                    )
                    cls._log_rejected_upload(filename, result["issues"], "zip_archive_wrong_extension")
                    return result
            else:
                result["valid"] = False
                result["issues"].append(
                    "Could not detect file type from content - file may be corrupted or disguised"
                )
                cls._log_rejected_upload(filename, result["issues"], "undetectable_mime")
                return result

        # Verify detected MIME type matches expected MIME types for this extension
        if detected_mime not in expected_mimes:
            result["valid"] = False
            result["issues"].append(
                f"MIME type mismatch: file extension is '{ext}' (expects {expected_mimes}) "
                f"but content is '{detected_mime}'"
            )
            cls._log_rejected_upload(filename, result["issues"], "mime_mismatch")
            return result

        # If claimed content type was provided, verify it matches
        if claimed_content_type:
            # Normalize content type (remove charset and other parameters)
            claimed_base = claimed_content_type.split(";")[0].strip().lower()
            if claimed_base not in expected_mimes and claimed_base != detected_mime:
                result["issues"].append(
                    f"Warning: Claimed Content-Type '{claimed_base}' differs from detected '{detected_mime}'"
                )
                # This is a warning, not a rejection - detected content is authoritative

        return result

    @classmethod
    def _log_rejected_upload(
        cls,
        filename: str,
        issues: List[str],
        rejection_type: str
    ) -> None:
        """Log rejected file uploads for security monitoring."""
        cls._logger.warning(
            "File upload rejected: %s - %s (%s)",
            filename,
            rejection_type,
            "; ".join(issues),
            extra={
                "event_type": "file_upload_rejected",
                "upload_filename": filename,  # Renamed to avoid LogRecord conflict
                "rejection_type": rejection_type,
                "issues": issues,
                "security_event": True
            }
        )


class SecurityScanner:
    """Security scanning and vulnerability detection"""

    @staticmethod
    def scan_file_upload(file_content: bytes, filename: str, content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive file upload security scan including MIME type validation.

        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename
            content_type: Content-Type header from the upload request (optional)

        Returns:
            Dict with scan results including safety status, detected types, and issues
        """
        results = {
            "safe": True,
            "issues": [],
            "file_type": "unknown",
            "detected_mime": None,
            "extension": None,
            "size": len(file_content),
            "mime_validation": None
        }

        # Check file size first
        if len(file_content) > SecurityConfig.MAX_FILE_SIZE:
            results["safe"] = False
            results["issues"].append(
                f"File size ({len(file_content)} bytes) exceeds limit ({SecurityConfig.MAX_FILE_SIZE} bytes)"
            )

        # Extract extension
        _, ext = os.path.splitext(filename.lower())
        results["extension"] = ext
        results["file_type"] = ext.lstrip(".") if ext else "unknown"

        # Perform MIME type validation
        mime_validation = FileUploadValidator.validate_mime_type(
            file_content, filename, content_type
        )
        results["mime_validation"] = mime_validation
        results["detected_mime"] = mime_validation.get("detected_mime")

        if not mime_validation["valid"]:
            results["safe"] = False
            results["issues"].extend(mime_validation["issues"])

        # Check for suspicious patterns (malware signatures)
        suspicious_patterns = [
            (b"<script", "JavaScript script tag"),
            (b"javascript:", "JavaScript protocol"),
            (b"vbscript:", "VBScript protocol"),
            (b"onload=", "Event handler injection"),
            (b"onerror=", "Event handler injection"),
            (b"eval(", "Eval function call"),
            (b"exec(", "Exec function call"),
            (b"MZ", "Windows executable signature"),  # PE/EXE files
            (b"\x7fELF", "Linux executable signature"),  # ELF files
            (b"#!/", "Shell script shebang"),
            (b"<?php", "PHP code"),
            (b"<%", "ASP/JSP code"),
        ]

        # Use lowercase for pattern matching in text content
        content_lower = file_content.lower()

        for pattern, description in suspicious_patterns:
            pattern_lower = pattern.lower() if isinstance(pattern, bytes) else pattern
            # Check both original and lowercase
            if pattern in file_content or pattern_lower in content_lower:
                # Special handling: Don't flag shebang in text files that legitimately could have it
                if pattern == b"#!/" and ext in [".sh", ".py", ".rb"]:
                    continue
                results["safe"] = False
                results["issues"].append(f"Suspicious pattern detected: {description}")

        # Additional check: Double extension attack (e.g., file.pdf.exe)
        base_name = os.path.basename(filename)
        if base_name.count(".") > 1:
            # Extract all extensions
            parts = base_name.split(".")
            if len(parts) > 2:
                suspicious_exts = [".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".hta"]
                for part in parts[1:-1]:  # Check intermediate "extensions"
                    if f".{part.lower()}" in suspicious_exts:
                        results["safe"] = False
                        results["issues"].append(
                            f"Potential double extension attack detected: filename contains '{part}'"
                        )

        # Log if file was rejected
        if not results["safe"]:
            scan_logger = logging.getLogger("security.file_upload")
            scan_logger.warning(
                "File upload scan failed: %s (%s)",
                filename,
                "; ".join(results["issues"]),
                extra={
                    "event_type": "file_scan_failed",
                    "upload_filename": filename,  # Renamed to avoid LogRecord conflict
                    "size": len(file_content),
                    "detected_mime": results["detected_mime"],
                    "issues": results["issues"],
                    "security_event": True
                }
            )

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
file_upload_validator = FileUploadValidator()


def validate_file_upload(
    file_content: bytes,
    filename: str,
    content_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for validating file uploads.
    Combines MIME type validation with security scanning.

    Args:
        file_content: Raw bytes of the uploaded file
        filename: Original filename with extension
        content_type: Content-Type header from upload (optional)

    Returns:
        Dict with comprehensive validation results

    Example:
        >>> result = validate_file_upload(file_bytes, "report.pdf", "application/pdf")
        >>> if not result["safe"]:
        ...     raise HTTPException(400, detail=result["issues"])
    """
    return security_scanner.scan_file_upload(file_content, filename, content_type)
