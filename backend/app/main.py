"""FastAPI application for MoodMentor (Milestone 1).

Routes only receive requests and return responses. All analysis logic
lives in sentiment.py.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
)
from .sentiment import EmptyTextError, NLPResourceError, analyze_batch, analyze_text

# Read settings from backend/.env if the file exists.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moodmentor")

APP_VERSION = "1.0.0"


def get_allowed_origins():
    """Frontend addresses allowed to call this API (comma-separated in .env)."""
    raw = os.getenv("FRONTEND_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="MoodMentor API",
    description=(
        "Baseline text-sentiment analysis (Milestone 1). "
        "Scores are produced by VADER and are not a clinical diagnosis."
    ),
    version=APP_VERSION,
)

# Only the local Streamlit frontend may call the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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


@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check that the API is running."""
    return {"status": "ok", "service": "MoodMentor API", "version": APP_VERSION}


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