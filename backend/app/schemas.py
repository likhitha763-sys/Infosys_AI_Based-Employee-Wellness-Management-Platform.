"""Request and response models (schemas) for the MoodMentor API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

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


class AnalyzeResponse(BaseModel):
    """The result for one analyzed text."""

    original_text: str
    processed_text: str
    positive_score: float = Field(..., ge=0, le=1)
    negative_score: float = Field(..., ge=0, le=1)
    neutral_score: float = Field(..., ge=0, le=1)
    compound_score: float = Field(..., ge=-1, le=1)
    sentiment: Literal["Positive", "Negative", "Neutral"]


class BatchAnalyzeResponse(BaseModel):
    """The results for a batch, in the same order as the request."""

    count: int
    results: list[AnalyzeResponse]


class HealthResponse(BaseModel):
    """Response of the health-check endpoint."""

    status: str
    service: str
    version: str