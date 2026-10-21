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
ANALYZE_URL = f"{API_BASE_URL}/api/analyze"
BATCH_URL = f"{API_BASE_URL}/api/analyze/batch"
HISTORY_URL = f"{API_BASE_URL}/api/history"

REQUEST_TIMEOUT = 30  # seconds
BATCH_TIMEOUT = 120  # seconds
MAX_BATCH_SIZE = 500  # entries per request (must match the backend)
BACKEND_START_COMMAND = "python -m uvicorn backend.app.main:app --reload --port 8000"

# For these status codes the backend sends a message meant for the user.
MESSAGE_STATUSES = {400, 401, 409, 500, 503}


class SessionExpiredError(Exception):
    """Raised when the backend rejects the login token (missing or expired)."""


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


def send_request(method, url, token=None, payload=None, params=None, timeout=REQUEST_TIMEOUT):
    """Send a request to the backend. Returns (data, error_message); one is None.

    Raises SessionExpiredError if a login token was sent and was rejected.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.request(
            method, url, json=payload, params=params, headers=headers, timeout=timeout
        )
    except requests.exceptions.ConnectionError:
        return None, (
            f"Cannot reach the backend at {API_BASE_URL}. "
            f"Please start it and try again. Command: {BACKEND_START_COMMAND}"
        )
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond. Please try again."
    except requests.exceptions.RequestException:
        return None, "Something went wrong while contacting the backend."

    if response.status_code == 401 and token:
        raise SessionExpiredError()

    if response.status_code in (200, 201):
        try:
            return response.json(), None
        except ValueError:
            return None, "The backend returned a response that could not be read."

    return None, get_error_message(response)


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


def register_user(full_name, email, password):
    """Create an account. Returns (user, error_message)."""
    payload = {"full_name": full_name, "email": email, "password": password}
    return send_request("POST", REGISTER_URL, payload=payload)


def login_user(email, password):
    """Log in. Returns ({access_token, token_type, user}, error_message)."""
    return send_request("POST", LOGIN_URL, payload={"email": email, "password": password})


# ---------------------------------------------------------------------------
# Analysis (login required)
# ---------------------------------------------------------------------------


def analyze_text(token, text):
    """Analyze one text; the backend saves the result. Returns (result, error)."""
    return send_request("POST", ANALYZE_URL, token=token, payload={"text": text})


def analyze_batch(token, texts):
    """Analyze many texts in chunks the backend accepts.

    Returns (list_of_results, error_message).
    """
    all_results = []
    for start in range(0, len(texts), MAX_BATCH_SIZE):
        chunk = texts[start : start + MAX_BATCH_SIZE]
        data, error = send_request(
            "POST", BATCH_URL, token=token, payload={"texts": chunk}, timeout=BATCH_TIMEOUT
        )
        if error:
            return None, error
        all_results.extend(data["results"])
    return all_results, None


def get_history(token, limit, offset):
    """Fetch one page of the user's saved analyses. Returns ({total, items}, error)."""
    return send_request(
        "GET", HISTORY_URL, token=token, params={"limit": limit, "offset": offset}
    )