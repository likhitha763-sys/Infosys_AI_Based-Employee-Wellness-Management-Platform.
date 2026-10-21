"""Input checks for the account forms.

This module contains no Streamlit code, so it can be tested on its own.
The backend repeats these checks; they are here to give quick, friendly
feedback before anything is sent.
"""

import re

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
MAX_NAME_LENGTH = 100
MAX_EMAIL_LENGTH = 255

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email):
    """A simple shape check: something@something.something, no spaces."""
    email = email.strip()
    return len(email) <= MAX_EMAIL_LENGTH and bool(EMAIL_PATTERN.match(email))


def password_problem(password):
    """Return what is wrong with a new password, or None if it is acceptable."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
    if len(password) > MAX_PASSWORD_LENGTH:
        return f"Password must be at most {MAX_PASSWORD_LENGTH} characters long."
    if not any(character.isalpha() for character in password):
        return "Password must contain at least one letter."
    if not any(character.isdigit() for character in password):
        return "Password must contain at least one number."
    return None


def validate_registration(full_name, email, password, confirm_password):
    """Return an error message for the registration form, or None if it is fine."""
    if not full_name.strip() or not email.strip() or not password or not confirm_password:
        return "Please fill in all the fields."
    if len(full_name.strip()) > MAX_NAME_LENGTH:
        return f"Name must be at most {MAX_NAME_LENGTH} characters."
    if not is_valid_email(email):
        return "Please enter a valid email address."
    problem = password_problem(password)
    if problem:
        return problem
    if password != confirm_password:
        return "The two passwords do not match."
    return None


def validate_login(email, password):
    """Return an error message for the login form, or None if it is fine."""
    if not email.strip() or not password:
        return "Please enter your email and password."
    if not is_valid_email(email):
        return "Please enter a valid email address."
    return None