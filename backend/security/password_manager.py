"""
Password Management Module

Uses bcrypt via passlib for password hashing (work factor 12). Retains
backward-compatible verification of legacy PBKDF2-HMAC-SHA256 hashes
(format: ``{hex_salt}${hex_hash}``). New hashes always use bcrypt.
"""

import re
import secrets
import hashlib
from typing import Optional

from passlib.context import CryptContext

# Primary: bcrypt. Legacy: PBKDF2 custom format (auto-detected in verify).
_crypt_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Legacy PBKDF2 settings (read-only, for verification of old hashes)
_LEGACY_ITERATIONS = 100000


class PasswordManager:
    """Password management with bcrypt hashing and policy enforcement."""

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt (work factor 12)."""
        return _crypt_ctx.hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash.

        Supports both bcrypt hashes (``$2b$...``) and legacy PBKDF2
        hashes (``{hex_salt}${hex_hash}``).
        """
        if hashed.startswith("$2b$") or hashed.startswith("$2a$"):
            return _crypt_ctx.verify(password, hashed)
        # Legacy PBKDF2-HMAC-SHA256 fallback
        if "$" not in hashed:
            return False
        salt, stored_hash = hashed.split("$", 1)
        computed = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), _LEGACY_ITERATIONS
        )
        return computed.hex() == stored_hash

    def needs_rehash(self, hashed: str) -> bool:
        """Check if hash should be upgraded to current algorithm."""
        if not (hashed.startswith("$2b$") or hashed.startswith("$2a$")):
            return True  # Legacy PBKDF2 → upgrade to bcrypt
        return _crypt_ctx.needs_update(hashed)

    def generate_password(self, length: int = 16) -> str:
        """Generate secure random password meeting policy requirements."""
        upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lower = "abcdefghijklmnopqrstuvwxyz"
        digits = "0123456789"
        special = "!@#$%^&*()"
        all_chars = upper + lower + digits + special
        # Guarantee at least one from each category
        password = [
            secrets.choice(upper),
            secrets.choice(lower),
            secrets.choice(digits),
            secrets.choice(special),
        ]
        password.extend(secrets.choice(all_chars) for _ in range(length - 4))
        # Shuffle to avoid predictable prefix
        result = list(password)
        for i in range(len(result) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            result[i], result[j] = result[j], result[i]
        return "".join(result)

    def check_password_strength(self, password: str) -> dict:
        """Check password strength with detailed scoring."""
        has_upper = bool(re.search(r"[A-Z]", password))
        has_lower = bool(re.search(r"[a-z]", password))
        has_digit = bool(re.search(r"\d", password))
        has_special = bool(re.search(r"[^A-Za-z0-9]", password))
        criteria_met = sum([has_upper, has_lower, has_digit, has_special])
        length = len(password)

        if length >= 16 and criteria_met >= 4:
            score = 5
        elif length >= 12 and criteria_met >= 3:
            score = 4
        elif length >= 10 and criteria_met >= 3:
            score = 3
        elif length >= 8 and criteria_met >= 2:
            score = 2
        else:
            score = 1

        return {
            "score": score,
            "length": length,
            "has_upper": has_upper,
            "has_lower": has_lower,
            "has_digit": has_digit,
            "has_special": has_special,
        }

    def validate_password_policy(self, password: str) -> bool:
        """Validate password meets policy requirements.

        Policy: >= 10 chars, must contain uppercase, lowercase, digit,
        and special character.
        """
        if len(password) < 10:
            return False
        strength = self.check_password_strength(password)
        return all([
            strength["has_upper"],
            strength["has_lower"],
            strength["has_digit"],
            strength["has_special"],
        ])
