"""Saving and reading analysis results (database logic).

This module contains no web code. FastAPI (main.py) calls these functions.
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Analysis

# Fields every result has.
REQUIRED_FIELDS = (
    "original_text",
    "processed_text",
    "positive_score",
    "negative_score",
    "neutral_score",
    "compound_score",
    "sentiment",
)

# Fields that may be missing (emotion detection is optional).
OPTIONAL_FIELDS = ("predicted_emotion", "emotion_confidence")


def save_analyses(db: Session, user_id, results):
    """Save several analysis results for a user in one transaction."""
    records = []
    for result in results:
        values = {field: result[field] for field in REQUIRED_FIELDS}
        values.update({field: result.get(field) for field in OPTIONAL_FIELDS})
        records.append(Analysis(user_id=user_id, **values))
    db.add_all(records)
    db.commit()
    return records


def list_analyses(db: Session, user_id, limit=20, offset=0):
    """Return a user's saved analyses, newest first."""
    query = (
        select(Analysis)
        .where(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(query))


def count_analyses(db: Session, user_id):
    """Return how many analyses a user has saved."""
    query = select(func.count()).select_from(Analysis).where(Analysis.user_id == user_id)
    return db.scalar(query)