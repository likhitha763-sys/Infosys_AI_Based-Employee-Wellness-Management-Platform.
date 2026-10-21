"""Checks for frontend/validation.py (the account form rules).

Run from the project root:
    python -m frontend.tests.check_validation
"""

import sys

from frontend.validation import (
    is_valid_email,
    password_problem,
    validate_login,
    validate_registration,
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


def says(message, fragment):
    """True if the message exists and contains the fragment (ignoring case)."""
    return message is not None and fragment in message.lower()


print("1. Email addresses")
check("a normal email is valid", is_valid_email("a@b.co"))
check("'abc' is not valid", not is_valid_email("abc"))
check("'a@b' (no dot) is not valid", not is_valid_email("a@b"))
check("an email with a space is not valid", not is_valid_email("a b@c.com"))
check("spaces around a valid email are ignored", is_valid_email("  a@b.co  "))
check("an email over 255 characters is not valid", not is_valid_email("a" * 250 + "@b.com"))

print("\n2. Passwords")
check("'GoodPass123' is acceptable", password_problem("GoodPass123") is None)
check("'abc12' is too short", says(password_problem("abc12"), "at least 8"))
check("'onlyletters' needs a number", says(password_problem("onlyletters"), "number"))
check("'12345678' needs a letter", says(password_problem("12345678"), "letter"))
check("130 characters is too long", says(password_problem("a1" * 65), "at most"))

print("\n3. Registration form")
good = ("Test Student", "student@example.com", "GoodPass123", "GoodPass123")
check("a correct form has no problem", validate_registration(*good) is None)
check("a blank name is rejected", says(validate_registration("   ", *good[1:]), "fill in"))
check("an empty confirm field is rejected", says(validate_registration(*good[:3], ""), "fill in"))
check("an invalid email is rejected", says(validate_registration(good[0], "abc", *good[2:]), "valid email"))
check("a weak password is rejected", says(validate_registration(good[0], good[1], "short1", "short1"), "at least 8"))
check("different passwords are rejected", says(validate_registration(*good[:3], "GoodPass124"), "do not match"))
check("a name over 100 characters is rejected", says(validate_registration("a" * 101, *good[1:]), "at most 100"))

print("\n4. Login form")
check("a correct form has no problem", validate_login("student@example.com", "anything") is None)
check("an empty password is rejected", says(validate_login("student@example.com", ""), "email and password"))
check("an invalid email is rejected", says(validate_login("abc", "anything"), "valid email"))

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)