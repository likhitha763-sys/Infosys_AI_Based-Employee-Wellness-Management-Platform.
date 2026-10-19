"""Checks for backend/app/security.py (password hashing and login tokens).

Run from the project root:
    python -m backend.tests.check_security
"""

import os
import sys
from unittest.mock import patch

import jwt

from backend.app.security import (
    SecurityConfigError,
    TokenError,
    WeakPasswordError,
    create_access_token,
    decode_access_token,
    hash_password,
    validate_password_strength,
    verify_password,
)

passed = 0
failed = 0


def check(description, condition):
    """Print PASS or FAIL for one check and keep count."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {description}")
    else:
        failed += 1
        print(f"  FAIL  {description}")


def raises(error_type, function, *arguments):
    """True if calling the function raises the given error type."""
    try:
        function(*arguments)
    except error_type:
        return True
    except Exception:
        return False
    return False


print("1. Password hashing")
first = hash_password("GoodPass123")
second = hash_password("GoodPass123")
check("the hash is not the password", "GoodPass123" not in first)
check("the hash uses Argon2", first.startswith("$argon2"))
check("the same password gives different hashes (random salt)", first != second)
check("the correct password verifies", verify_password("GoodPass123", first))
check("a wrong password is rejected", not verify_password("WrongPass123", first))
check("a damaged hash is rejected, not a crash", not verify_password("GoodPass123", "garbage"))
check("non-English characters work", verify_password("pässwörd9", hash_password("pässwörd9")))

print("\n2. Password rules")
check("accepts 'GoodPass123'", not raises(WeakPasswordError, validate_password_strength, "GoodPass123"))
check("rejects too short ('abc12')", raises(WeakPasswordError, validate_password_strength, "abc12"))
check("rejects no number ('onlyletters')", raises(WeakPasswordError, validate_password_strength, "onlyletters"))
check("rejects no letter ('12345678')", raises(WeakPasswordError, validate_password_strength, "12345678"))
check("rejects over 128 characters", raises(WeakPasswordError, validate_password_strength, "a1" * 65))

print("\n3. Login tokens")
with patch.dict(os.environ, {"SECRET_KEY": "k" * 64}):
    token = create_access_token(42)
    check("a valid token returns the user id", decode_access_token(token) == 42)

    parts = token.split(".")
    parts[1] = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
    check("a tampered token is rejected", raises(TokenError, decode_access_token, ".".join(parts)))

    expired = create_access_token(42, expires_minutes=-1)
    check("an expired token is rejected", raises(TokenError, decode_access_token, expired))

    foreign = jwt.encode({"sub": "42", "exp": 9999999999}, "x" * 64, algorithm="HS256")
    check("a token signed with another key is rejected", raises(TokenError, decode_access_token, foreign))

    check("random text is rejected", raises(TokenError, decode_access_token, "not.a.token"))
    check("an empty token is rejected", raises(TokenError, decode_access_token, ""))

print("\n4. Secret key safety")
with patch.dict(os.environ, {"SECRET_KEY": ""}):
    check("a missing SECRET_KEY is refused", raises(SecurityConfigError, create_access_token, 1))
with patch.dict(os.environ, {"SECRET_KEY": "change-me"}):
    check("the example value 'change-me' is refused", raises(SecurityConfigError, create_access_token, 1))
with patch.dict(os.environ, {"SECRET_KEY": "short"}):
    check("a short SECRET_KEY is refused", raises(SecurityConfigError, create_access_token, 1))

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)