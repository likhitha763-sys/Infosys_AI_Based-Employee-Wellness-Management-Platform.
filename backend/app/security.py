"""Password hashing and login tokens for MoodMentor.

This module contains no web code. FastAPI (main.py) calls these functions.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from dotenv import load_dotenv

# Read settings from backend/.env if the file exists.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ALGORITHM = "HS256"
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
MIN_SECRET_KEY_LENGTH = 32


class SecurityConfigError(RuntimeError):
    """Raised when SECRET_KEY is missing or too weak."""


class WeakPasswordError(ValueError):
    """Raised when a new password does not meet the rules."""


class TokenError(Exception):
    """Raised when a login token is missing, expired, or invalid."""


_password_hasher = PasswordHasher()


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def validate_password_strength(password):
    """Check the rules for a new password. Raises WeakPasswordError."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPasswordError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise WeakPasswordError(
            f"Password must be at most {MAX_PASSWORD_LENGTH} characters long."
        )
    if not any(character.isalpha() for character in password):
        raise WeakPasswordError("Password must contain at least one letter.")
    if not any(character.isdigit() for character in password):
        raise WeakPasswordError("Password must contain at least one number.")


def hash_password(password):
    """Return a salted one-way hash of the password."""
    return _password_hasher.hash(password)


def verify_password(password, password_hash):
    """True if the password matches the stored hash, otherwise False."""
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


# ---------------------------------------------------------------------------
# Login tokens
# ---------------------------------------------------------------------------


def _get_secret_key():
    """Read SECRET_KEY from the environment and refuse weak values."""
    secret_key = os.getenv("SECRET_KEY", "")
    if not secret_key or secret_key == "change-me":
        raise SecurityConfigError(
            "SECRET_KEY is not set. Add a long random value to backend/.env."
        )
    if len(secret_key) < MIN_SECRET_KEY_LENGTH:
        raise SecurityConfigError(
            f"SECRET_KEY is too short. Use at least {MIN_SECRET_KEY_LENGTH} characters."
        )
    return secret_key


def create_access_token(user_id, expires_minutes=None):
    """Create a signed login token for a user."""
    if expires_minutes is None:
        try:
            expires_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
        except ValueError:
            expires_minutes = 60

    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, _get_secret_key(), algorithm=ALGORITHM)


def decode_access_token(token):
    """Return the user id inside a valid token. Raises TokenError otherwise."""
    secret_key = _get_secret_key()
    try:
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
        return int(payload["sub"])
    except (jwt.PyJWTError, ValueError, TypeError) as exc:
        raise TokenError("Invalid or expired login token.") from exc
    
    

def check_secret_key():
    """Raise SecurityConfigError if SECRET_KEY is missing or weak.

    Called when the API starts, so a bad setting is noticed immediately.
    """
    _get_secret_key()