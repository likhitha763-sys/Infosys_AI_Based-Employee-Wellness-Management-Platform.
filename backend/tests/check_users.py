"""Checks for backend/app/users.py using a temporary in-memory database.

Run from the project root:
    python -m backend.tests.check_users
"""

import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app import models  # noqa: F401  (registers the tables)
from backend.app.database import Base
from backend.app.security import WeakPasswordError
from backend.app.users import (
    EmailAlreadyRegisteredError,
    authenticate_user,
    create_user,
    get_user_by_email,
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


engine = create_engine("sqlite:///:memory:")  # temporary, disappears afterwards
Base.metadata.create_all(engine)

with Session(engine) as db:
    print("1. Creating a user")
    user = create_user(db, "  Test Student  ", " Student@Example.com ", "GoodPass123")
    check("the user gets an id", user.id is not None)
    check("the name is trimmed", user.full_name == "Test Student")
    check("the email is lowercased and trimmed", user.email == "student@example.com")
    check("the stored hash is Argon2", user.password_hash.startswith("$argon2"))
    check("the plain password is not stored", "GoodPass123" not in user.password_hash)

    print("\n2. Duplicate emails")
    check(
        "the same email in another case is rejected",
        raises(EmailAlreadyRegisteredError, create_user, db, "Other", "STUDENT@example.com", "GoodPass123"),
    )

    print("\n3. Weak passwords")
    check(
        "a weak password is rejected",
        raises(WeakPasswordError, create_user, db, "Weak", "weak@example.com", "short1"),
    )
    check("no user is created for it", get_user_by_email(db, "weak@example.com") is None)

    print("\n4. Logging in")
    check(
        "the correct password returns the user",
        getattr(authenticate_user(db, "student@example.com", "GoodPass123"), "id", None) == user.id,
    )
    check(
        "the email case does not matter",
        getattr(authenticate_user(db, "STUDENT@EXAMPLE.COM", "GoodPass123"), "id", None) == user.id,
    )
    check("a wrong password returns None", authenticate_user(db, "student@example.com", "WrongPass123") is None)
    check("an unknown email returns None", authenticate_user(db, "nobody@example.com", "GoodPass123") is None)

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)