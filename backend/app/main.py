"""FastAPI application for MoodMentor.

Routes only receive requests and return responses. Sentiment logic lives in
sentiment.py and account logic in users.py.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .database import create_tables, get_db
from .models import User
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from .security import (
    SecurityConfigError,
    TokenError,
    WeakPasswordError,
    check_secret_key,
    create_access_token,
    decode_access_token,
)
from .sentiment import EmptyTextError, NLPResourceError, analyze_batch, analyze_text
from .users import (
    EmailAlreadyRegisteredError,
    authenticate_user,
    create_user,
    get_user_by_id,
)

# Read settings from backend/.env if the file exists.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moodmentor")

APP_VERSION = "1.1.0"


@asynccontextmanager
async def lifespan(app):
    """Runs once when the API starts."""
    create_tables()
    try:
        check_secret_key()
    except SecurityConfigError as exc:
        logger.error("Login is disabled until this is fixed: %s", exc)
    yield


def get_allowed_origins():
    """Frontend addresses allowed to call this API (comma-separated in .env)."""
    raw = os.getenv("FRONTEND_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="MoodMentor API",
    description=(
        "Text-sentiment analysis with user accounts. "
        "Scores are produced by VADER and are not a clinical diagnosis."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# Only the local Streamlit frontend may call the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.exception_handler(RequestValidationError)
async def handle_invalid_request(request, exc):
    """Return a short, safe message instead of FastAPI's long default."""
    problems = []
    for error in exc.errors()[:3]:
        location = ".".join(str(part) for part in error["loc"] if part != "body")
        message = error["msg"]
        problems.append(f"{location}: {message}" if location else message)
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request. " + "; ".join(problems)},
    )


def run_analysis(function, *arguments):
    """Run an analysis function and turn its errors into HTTP errors."""
    try:
        return function(*arguments)
    except EmptyTextError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NLPResourceError as exc:
        logger.error("Missing NLP resource: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        # Full details go to the server log only, never to the user.
        logger.exception("Unexpected error while analyzing text")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while analyzing the text. Please try again.",
        )


# ---------------------------------------------------------------------------
# Login check (used by routes that need a logged-in user)
# ---------------------------------------------------------------------------

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
):
    """Return the logged-in user, or raise 401."""
    not_logged_in = HTTPException(
        status_code=401,
        detail="You are not logged in, or your session has expired. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise not_logged_in

    try:
        user_id = decode_access_token(credentials.credentials)
    except TokenError:
        raise not_logged_in
    except SecurityConfigError as exc:
        logger.error("Security configuration problem: %s", exc)
        raise HTTPException(
            status_code=500, detail="The server is not configured correctly."
        )

    user = get_user_by_id(db, user_id)
    if user is None:
        raise not_logged_in
    return user


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check that the API is running."""
    return {"status": "ok", "service": "MoodMentor API", "version": APP_VERSION}


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


@app.post(
    "/api/auth/register",
    response_model=UserResponse,
    status_code=201,
    tags=["Accounts"],
)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new account."""
    try:
        return create_user(db, request.full_name, request.email, request.password)
    except WeakPasswordError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EmailAlreadyRegisteredError:
        raise HTTPException(
            status_code=409, detail="An account with this email already exists."
        )
    except Exception:
        logger.exception("Unexpected error while registering a user")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while creating the account. Please try again.",
        )


@app.post("/api/auth/login", response_model=TokenResponse, tags=["Accounts"])
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Check the email and password and return a login token."""
    try:
        user = authenticate_user(db, request.email, request.password)
    except Exception:
        logger.exception("Unexpected error during login")
        raise HTTPException(
            status_code=500, detail="Something went wrong. Please try again."
        )

    if user is None:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    try:
        token = create_access_token(user.id)
    except SecurityConfigError as exc:
        logger.error("Cannot create login token: %s", exc)
        raise HTTPException(
            status_code=500, detail="Login is temporarily unavailable. Please try again later."
        )

    return {"access_token": token, "token_type": "bearer", "user": user}


@app.get("/api/auth/me", response_model=UserResponse, tags=["Accounts"])
def read_current_user(current_user: User = Depends(get_current_user)):
    """Return the logged-in user. Requires a login token."""
    return current_user


# ---------------------------------------------------------------------------
# Sentiment analysis (still public in this step)
# ---------------------------------------------------------------------------


@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Sentiment"])
def analyze(request: AnalyzeRequest):
    """Analyze the sentiment of one piece of text."""
    return run_analysis(analyze_text, request.text)


@app.post(
    "/api/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Sentiment"]
)
def analyze_multiple(request: BatchAnalyzeRequest):
    """Analyze several texts in one request (used for TXT and CSV uploads)."""
    results = run_analysis(analyze_batch, request.texts)
    return {"count": len(results), "results": results}