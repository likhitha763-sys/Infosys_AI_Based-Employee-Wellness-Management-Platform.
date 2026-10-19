"""User accounts: creating users and checking logins (database logic).

This module contains no web code. FastAPI (main.py) calls these functions.
"""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import User
from .security import hash_password, validate_password_strength, verify_password


class EmailAlreadyRegisteredError(Exception):
    """Raised when an account with this email already exists."""


# A hash of a throwaway password. Checking a login for an unknown email
# against it takes about as long as a real check, so the response time does
# not reveal which emails are registered.
_DUMMY_HASH = hash_password("throwaway-password-1")


def normalize_email(email):
    """Emails are compared in lowercase without surrounding spaces."""
    return email.strip().lower()


def get_user_by_email(db: Session, email):
    """Return the user with this email, or None."""
    return db.scalar(select(User).where(User.email == normalize_email(email)))


def get_user_by_id(db: Session, user_id):
    """Return the user with this id, or None."""
    return db.get(User, user_id)


def create_user(db: Session, full_name, email, password):
    """Create and return a new user.

    Raises WeakPasswordError or EmailAlreadyRegisteredError.
    """
    validate_password_strength(password)
    email = normalize_email(email)

    if get_user_by_email(db, email) is not None:
        raise EmailAlreadyRegisteredError("An account with this email already exists.")

    user = User(
        full_name=full_name.strip(),
        email=email,
        password_hash=hash_password(password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Two people registered the same email at the same moment.
        db.rollback()
        raise EmailAlreadyRegisteredError("An account with this email already exists.")
    db.refresh(user)
    return user


def authenticate_user(db: Session, email, password):
    """Return the user if the email and password are correct, otherwise None."""
    user = get_user_by_email(db, email)
    if user is None:
        verify_password(password, _DUMMY_HASH)  # keep the timing similar
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user