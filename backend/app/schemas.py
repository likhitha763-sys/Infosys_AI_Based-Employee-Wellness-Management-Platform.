"""Request and response models (schemas) for the MoodMentor API."""

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from .emotion import describe_emotion

MAX_TEXT_LENGTH = 10_000  # characters allowed in a single entry
MAX_BATCH_SIZE = 500  # entries allowed in one batch request


class AnalyzeRequest(BaseModel):
    """One text sent by the frontend."""

    text: str = Field(
        ...,
        max_length=MAX_TEXT_LENGTH,
        description="The journal entry, message, or paragraph to analyze.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "I am very happy today. Everything is going well!"}]
        }
    }


class BatchAnalyzeRequest(BaseModel):
    """Several texts sent by the frontend (for example, rows of a CSV file)."""

    texts: list[Annotated[str, Field(max_length=MAX_TEXT_LENGTH)]] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="The entries to analyze, one string per entry.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texts": [
                        "I am very happy today. Everything is going well!",
                        "I feel very sad and disappointed today.",
                    ]
                }
            ]
        }
    }


class EmotionScore(BaseModel):
    """One emotion and the model's score for it."""

    emotion: str
    score: float = Field(..., ge=0, le=1)


class AnalyzeResponse(BaseModel):
    """The result for one analyzed text.

    Sentiment (VADER) and emotion (separate model) are different results.
    The emotion fields are empty when emotion detection is off or unavailable.
    """

    original_text: str
    processed_text: str
    positive_score: float = Field(..., ge=0, le=1)
    negative_score: float = Field(..., ge=0, le=1)
    neutral_score: float = Field(..., ge=0, le=1)
    compound_score: float = Field(..., ge=-1, le=1)
    sentiment: Literal["Positive", "Negative", "Neutral"]

    predicted_emotion: str | None = None
    emotion_name: str | None = None
    emotion_emoji: str | None = None
    emotion_confidence: float | None = Field(None, ge=0, le=1)
    top_emotions: list[EmotionScore] | None = None
    emotion_note: str | None = None


class BatchAnalyzeResponse(BaseModel):
    """The results for a batch, in the same order as the request."""

    count: int
    results: list[AnalyzeResponse]


class HealthResponse(BaseModel):
    """Response of the health-check endpoint."""

    status: str
    service: str
    version: str
    
    

class RegisterRequest(BaseModel):
    """Data sent when creating an account."""

    full_name: str = Field(..., max_length=100)
    email: EmailStr
    password: str = Field(..., max_length=128)

    @field_validator("full_name")
    @classmethod
    def name_must_not_be_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Name must not be empty.")
        return value

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "full_name": "Test Student",
                    "email": "student1@example.com",
                    "password": "GoodPass123",
                }
            ]
        }
    }


class LoginRequest(BaseModel):
    """Data sent when logging in."""

    email: EmailStr
    password: str = Field(..., max_length=128)

    model_config = {
        "json_schema_extra": {
            "examples": [{"email": "student1@example.com", "password": "GoodPass123"}]
        }
    }


class UserResponse(BaseModel):
    """A user as shown to the frontend. Never includes the password."""

    id: int
    full_name: str
    email: str

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Returned after a successful login."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse
    
    

class AnalysisRecord(AnalyzeResponse):
    """One saved analysis in a user's history."""

    id: int
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("created_at")
    @classmethod
    def assume_utc(cls, value):
        """SQLite returns times without a time zone; they are stored in UTC."""
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    @model_validator(mode="after")
    def add_emotion_display(self):
        """Saved rows only hold the emotion label; add its name and emoji."""
        if self.predicted_emotion and not self.emotion_name:
            self.emotion_name, self.emotion_emoji = describe_emotion(self.predicted_emotion)
        return self    


class HistoryResponse(BaseModel):
    """A page of a user's saved analyses, newest first."""

    total: int
    items: list[AnalysisRecord]