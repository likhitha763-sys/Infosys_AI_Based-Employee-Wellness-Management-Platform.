"""Talking to the MoodMentor backend: HTTP requests only, no screen code."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Read settings from frontend/.env if the file exists.
load_dotenv(Path(__file__).resolve().parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
HEALTH_URL = f"{API_BASE_URL}/health"
REGISTER_URL = f"{API_BASE_URL}/api/auth/register"
LOGIN_URL = f"{API_BASE_URL}/api/auth/login"

REQUEST_TIMEOUT = 30  # seconds
BACKEND_START_COMMAND = "python -m uvicorn backend.app.main:app --reload --port 8000"

# For these status codes the backend sends a message meant for the user.
MESSAGE_STATUSES = {400, 401, 409, 500, 503}


def backend_is_online():
    """Quick check of the backend health endpoint."""
    try:
        return requests.get(HEALTH_URL, timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_error_message(response):
    """Turn an error response from the backend into a friendly message."""
    if response.status_code in MESSAGE_STATUSES:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        if isinstance(detail, str) and detail:
            return detail
    if response.status_code == 422:
        return "Some of the details you entered were not accepted. Please check them and try again."
    return "The backend had a problem handling this request. Please try again."


def post_json(url, payload, token=None, timeout=REQUEST_TIMEOUT):
    """POST JSON to the backend. Returns (data, error_message); one is None."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return None, (
            f"Cannot reach the backend at {API_BASE_URL}. "
            f"Please start it and try again. Command: {BACKEND_START_COMMAND}"
        )
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond. Please try again."
    except requests.exceptions.RequestException:
        return None, "Something went wrong while contacting the backend."

    if response.status_code in (200, 201):
        try:
            return response.json(), None
        except ValueError:
            return None, "The backend returned a response that could not be read."

    return None, get_error_message(response)


def register_user(full_name, email, password):
    """Create an account. Returns (user, error_message)."""
    payload = {"full_name": full_name, "email": email, "password": password}
    return post_json(REGISTER_URL, payload)


def login_user(email, password):
    """Log in. Returns ({access_token, token_type, user}, error_message)."""
    return post_json(LOGIN_URL, {"email": email, "password": password})