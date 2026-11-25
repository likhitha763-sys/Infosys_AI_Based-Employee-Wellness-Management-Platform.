"""FastAPI application for MoodMentor.

Routes only receive requests and return responses. Sentiment logic lives in
sentiment.py and account logic in users.py.
"""

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .analyses import (
    count_analyses,
    delete_all_analyses,
    list_all_analyses,
    list_analyses,
    save_analyses,
)
from .dashboard import build_dashboard
from .database import create_tables, get_db
from .models import User
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    DashboardResponse,
    HealthResponse,
    HistoryResponse,
    LoginRequest,
    RegisterRequest,
    ResourcesResponse,
    TokenResponse,
    UserResponse,
    GoalCreate,
    GoalResponse,
    GoalsResponse,
    GoalUpdate,
)
from .security import (
    SecurityConfigError,
    TokenError,
    WeakPasswordError,
    check_secret_key,
    create_access_token,
    decode_access_token,
)
from .emotion import attach_emotions, last_engine,  warm_up
from .sentiment import EmptyTextError, NLPResourceError, analyze_batch, analyze_text
from .users import (
    EmailAlreadyRegisteredError,
    authenticate_user,
    create_user,
    get_user_by_id,
)
from .safety import check_urgent
from .goals import (
    GoalLimitError,
    GoalNotFoundError,
    create_goal,
    delete_all_goals,
    delete_goal,
    list_goals,
    set_goal_completed,
)
from .resources import get_resources

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
     # Load the emotion model in the background so the first request is fast.
    threading.Thread(target=warm_up, daemon=True).start()
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
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
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
# Sentiment analysis (login required; every result is saved)
# ---------------------------------------------------------------------------


def save_or_fail(db, user, results):
    """Save results for the user. Turns database errors into a safe 500."""
    try:
        save_analyses(db, user.id, results)
    except Exception:
        db.rollback()
        logger.exception("Could not save analysis results")
        raise HTTPException(
            status_code=500,
            detail=(
                "The text was analyzed, but the result could not be saved. "
                "Please try again."
            ),
        )

@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Sentiment"])
def analyze(
    request: AnalyzeRequest,
    save: bool = Query(True, description="Set to false to analyze without saving the result."),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyze one text. The result is saved to the user's history unless save=false."""
    result = run_analysis(analyze_text, request.text)
    [result] = attach_emotions([result])
    if save:
        save_or_fail(db, current_user, [result])
    result["emotion_engine"] = last_engine() if result.get("predicted_emotion") else None
    result["safety_alert"] = check_urgent(result["original_text"])
    return result


@app.post(
    "/api/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Sentiment"]
)
def analyze_multiple(
    request: BatchAnalyzeRequest,
    save: bool = Query(True, description="Set to false to analyze without saving the results."),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyze several texts. Results are saved unless save=false."""
    results = run_analysis(analyze_batch, request.texts)
    results = attach_emotions(results)
    if save:
        save_or_fail(db, current_user, results)
    engine = last_engine()
    for item in results:
        item["emotion_engine"] = engine if item.get("predicted_emotion") else None
        item["safety_alert"] = check_urgent(item["original_text"])
    return {"count": len(results), "results": results}

@app.get("/api/history", response_model=HistoryResponse, tags=["Sentiment"])
def history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the logged-in user's saved analyses, newest first."""
    try:
        return {
            "total": count_analyses(db, current_user.id),
            "items": list_analyses(db, current_user.id, limit, offset),
        }
    except Exception:
        logger.exception("Could not read analysis history")
        raise HTTPException(
            status_code=500,
            detail="Your history could not be loaded. Please try again.",
        )
        
        
@app.get("/api/history/export", response_model=HistoryResponse, tags=["Privacy"])
def export_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return every saved analysis of the logged-in user (a copy of their data)."""
    try:
        items = list_all_analyses(db, current_user.id)
        return {"total": len(items), "items": items}
    except Exception:
        logger.exception("Could not export analysis history")
        raise HTTPException(
            status_code=500,
            detail="Your data could not be exported. Please try again.",
        )


@app.delete("/api/history", tags=["Privacy"])
def delete_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Permanently delete every saved analysis of the logged-in user."""
    try:
        deleted = delete_all_analyses(db, current_user.id)
    except Exception:
        db.rollback()
        logger.exception("Could not delete analysis history")
        raise HTTPException(
            status_code=500,
            detail="Your saved analyses could not be deleted. Please try again.",
        )
    return {"deleted": deleted}
        
        

# ---------------------------------------------------------------------------
# Dashboard (login required)
# ---------------------------------------------------------------------------


@app.get("/api/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
def dashboard(
    rhythm_limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Summary numbers and chart data for the logged-in user's saved analyses."""
    try:
        return build_dashboard(db, current_user.id, rhythm_limit)
    except Exception:
        logger.exception("Could not build the dashboard")
        raise HTTPException(
            status_code=500,
            detail="Your dashboard could not be loaded. Please try again.",
        )

@app.get("/api/resources", response_model=ResourcesResponse, tags=["Resources"])
def read_resources(current_user: User = Depends(get_current_user)):
    """Fixed wellness resources and help contacts. Requires a login token."""
    return get_resources()     


# ---------------------------------------------------------------------------
# Wellness goals (login required)
# ---------------------------------------------------------------------------


@app.get("/api/goals", response_model=GoalsResponse, tags=["Goals"])
def read_goals(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the logged-in user's goals."""
    try:
        items = list_goals(db, current_user.id)
        return {"total": len(items), "items": items}
    except Exception:
        logger.exception("Could not read goals")
        raise HTTPException(
            status_code=500, detail="Your goals could not be loaded. Please try again."
        )


@app.post("/api/goals", response_model=GoalResponse, status_code=201, tags=["Goals"])
def add_goal(
    request: GoalCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a goal for the logged-in user."""
    try:
        return create_goal(db, current_user.id, request.title, request.target_date)
    except (ValueError, GoalLimitError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        logger.exception("Could not save a goal")
        raise HTTPException(
            status_code=500, detail="The goal could not be saved. Please try again."
        )


@app.patch("/api/goals/{goal_id}", response_model=GoalResponse, tags=["Goals"])
def update_goal(
    goal_id: int,
    request: GoalUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark one of the user's goals as done, or reopen it."""
    try:
        return set_goal_completed(db, current_user.id, goal_id, request.completed)
    except GoalNotFoundError:
        raise HTTPException(status_code=404, detail="Goal not found.")
    except Exception:
        db.rollback()
        logger.exception("Could not update a goal")
        raise HTTPException(
            status_code=500, detail="The goal could not be updated. Please try again."
        )


@app.delete("/api/goals/{goal_id}", tags=["Goals"])
def remove_goal(
    goal_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete one of the user's goals."""
    try:
        delete_goal(db, current_user.id, goal_id)
    except GoalNotFoundError:
        raise HTTPException(status_code=404, detail="Goal not found.")
    except Exception:
        db.rollback()
        logger.exception("Could not delete a goal")
        raise HTTPException(
            status_code=500, detail="The goal could not be deleted. Please try again."
        )
    return {"deleted": 1}


@app.delete("/api/goals", tags=["Goals"])
def remove_all_goals(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete every goal of the logged-in user."""
    try:
        deleted = delete_all_goals(db, current_user.id)
    except Exception:
        db.rollback()
        logger.exception("Could not delete goals")
        raise HTTPException(
            status_code=500, detail="Your goals could not be deleted. Please try again."
        )
    return {"deleted": deleted}   