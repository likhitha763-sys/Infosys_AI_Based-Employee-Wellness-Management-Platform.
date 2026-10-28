"""Numbers for the dashboard: summary cards and chart data (database logic).

This module contains no web code. Everything is calculated from the
logged-in user's saved analyses, and nothing is made up.
"""

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from .emotion import NO_CLEAR_EMOTION, describe_emotion
from .models import Analysis

SENTIMENTS = ("Positive", "Neutral", "Negative")
NOT_REAL_EMOTIONS = ("neutral", NO_CLEAR_EMOTION)  # not counted as a "detected emotion"
PREVIEW_LENGTH = 80


def percent(count, total):
    """count as a percentage of total, with one decimal (0.0 if total is 0)."""
    return round(100 * count / total, 1) if total else 0.0


def top_labels(counts):
    """Return the label(s) with the highest count, in alphabetical order.

    More than one label means they are tied.
    """
    if not counts:
        return []
    highest = max(counts.values())
    return sorted(label for label, count in counts.items() if count == highest)


def preview(text, length=PREVIEW_LENGTH):
    """Collapse extra whitespace and cut long text for display."""
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 3] + "..."


def make_emotion_count(label, count, total):
    """One row of the emotion counts."""
    name, emoji = describe_emotion(label)
    return {
        "emotion": label,
        "name": name,
        "emoji": emoji,
        "count": count,
        "percent": percent(count, total),
        "is_emotion": label not in NOT_REAL_EMOTIONS,
    }


def build_dashboard(db: Session, user_id, rhythm_limit=50):
    """Calculate the dashboard numbers for one user."""
    # Only this user's entries with a valid sentiment label are counted.
    valid = and_(Analysis.user_id == user_id, Analysis.sentiment.in_(SENTIMENTS))

    total, average = db.execute(
        select(func.count(Analysis.id), func.avg(Analysis.compound_score)).where(valid)
    ).one()

    # Sentiment counts: every category is listed, even with a count of 0.
    sentiment_totals = {label: 0 for label in SENTIMENTS}
    sentiment_rows = db.execute(
        select(Analysis.sentiment, func.count()).where(valid).group_by(Analysis.sentiment)
    ).all()
    for label, count in sentiment_rows:
        sentiment_totals[label] = count

    # Emotion counts: only entries that actually have a saved emotion.
    emotion_rows = db.execute(
        select(Analysis.predicted_emotion, func.count())
        .where(valid, Analysis.predicted_emotion.is_not(None), Analysis.predicted_emotion != "")
        .group_by(Analysis.predicted_emotion)
    ).all()
    emotion_totals = {label: count for label, count in emotion_rows}
    entries_with_emotion = sum(emotion_totals.values())

    emotion_counts = [
        make_emotion_count(label, count, entries_with_emotion)
        for label, count in emotion_totals.items()
    ]
    emotion_counts.sort(key=lambda item: (-item["count"], item["name"]))

    # "Most common emotion" only considers real emotions (not neutral / no clear emotion).
    real_counts = {item["emotion"]: item["count"] for item in emotion_counts if item["is_emotion"]}
    leaders = top_labels(real_counts)
    most_common_emotions = [item for item in emotion_counts if item["emotion"] in leaders]

    # The line chart: the newest entries, shown oldest first.
    newest = db.scalars(
        select(Analysis)
        .where(valid)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(rhythm_limit)
    )
    recent = list(newest)
    recent.reverse()
    first_number = total - len(recent) + 1  # numbered by position in the full history

    rhythm = []
    for offset, row in enumerate(recent):
        name = emoji = None
        if row.predicted_emotion:
            name, emoji = describe_emotion(row.predicted_emotion)
        rhythm.append(
            {
                "number": first_number + offset,
                "created_at": row.created_at,
                "compound_score": row.compound_score,
                "sentiment": row.sentiment,
                "text_preview": preview(row.original_text),
                "emotion_name": name,
                "emotion_emoji": emoji,
            }
        )

    return {
        "summary": {
            "total_entries": total,
            "average_compound": round(float(average), 4) if average is not None else None,
            "most_common_sentiments": top_labels(sentiment_totals) if total else [],
            "entries_with_emotion": entries_with_emotion,
            "most_common_emotions": most_common_emotions,
        },
        "sentiment_counts": [
            {
                "sentiment": label,
                "count": sentiment_totals[label],
                "percent": percent(sentiment_totals[label], total),
            }
            for label in SENTIMENTS
        ],
        "emotion_counts": emotion_counts,
        "rhythm": rhythm,
    }