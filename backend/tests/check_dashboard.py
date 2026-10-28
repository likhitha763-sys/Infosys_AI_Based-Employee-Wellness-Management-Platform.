"""Checks for the dashboard numbers (dashboard.py and GET /api/dashboard).

Uses a temporary in-memory database and made-up test rows, so the expected
numbers are known in advance. Run from the project root:
    python -m backend.tests.check_dashboard
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app import main, models
from backend.app.dashboard import build_dashboard, preview, top_labels
from backend.app.database import Base, get_db

passed = 0
failed = 0


def check(description, condition):
    """Print PASS or FAIL for one check and keep count."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {description}")
    else:
        failed += 1
        print(f"  FAIL  {description}")


# A temporary database that replaces the real one for this test only.
engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


main.app.dependency_overrides[get_db] = override_get_db
client = TestClient(main.app)
env_patch = patch.dict(os.environ, {"SECRET_KEY": "t" * 64, "EMOTION_ENABLED": "false"})
env_patch.start()


def register(email):
    """Create an account and return (user id, login headers)."""
    body = {"full_name": "Test", "email": email, "password": "GoodPass123"}
    client.post("/api/auth/register", json=body)
    token = client.post("/api/auth/login", json={"email": email, "password": "GoodPass123"}).json()["access_token"]
    headers = {"Authorization": "Bearer " + token}
    return client.get("/api/auth/me", headers=headers).json()["id"], headers


def add_rows(user_id, rows, first_minute=1):
    """Save made-up rows. Each row is (text, compound, sentiment, emotion)."""
    with TestingSession() as session:
        for offset, (text, compound, sentiment, emotion) in enumerate(rows):
            session.add(models.Analysis(
                user_id=user_id, original_text=text, processed_text=text.lower(),
                positive_score=0.0, negative_score=0.0, neutral_score=1.0,
                compound_score=compound, sentiment=sentiment, predicted_emotion=emotion,
                emotion_confidence=None if emotion is None else 0.8,
                created_at=datetime(2026, 10, 1, 10, first_minute + offset, tzinfo=timezone.utc),
            ))
        session.commit()


def dashboard_for(user_id, rhythm_limit=50):
    with TestingSession() as session:
        return build_dashboard(session, user_id, rhythm_limit)


# Alice's entries, oldest first. The last row has an invalid sentiment label
# and must be ignored everywhere.
ROWS = [
    ("I am very happy today.", 0.8, "Positive", "joy"),
    ("I feel very sad.", -0.6, "Negative", "sadness"),
    ("The meeting is at 10 AM.", 0.0, "Neutral", "neutral"),
    ("Great news about the project!", 0.5, "Positive", "joy"),
    ("An old entry saved before emotions.", -0.2, "Negative", None),
    ("I am excited about the trip.", 0.4, "Positive", "excitement"),
    ("Hmm, hard to say.", 0.0, "Neutral", "no_clear_emotion"),
]
scores = [row[1] for row in ROWS]

alice_id, alice = register("alice@example.com")
add_rows(alice_id, ROWS)
add_rows(alice_id, [("A broken record", 0.9, "Weird", "anger")], first_minute=30)
bob_id, bob = register("bob@example.com")
add_rows(bob_id, [("Bob's private note", 0.3, "Positive", "joy")], first_minute=5)

result = dashboard_for(alice_id)
summary = result["summary"]

print("1. Summary cards")
check("7 entries are counted (the broken record is left out)", summary["total_entries"] == 7)
check("the average is the average of the stored scores", summary["average_compound"] == round(sum(scores) / len(scores), 4))
check("the most common sentiment is Positive", summary["most_common_sentiments"] == ["Positive"])
check("6 entries have an emotion (the old entry has none)", summary["entries_with_emotion"] == 6)
check("the most common detected emotion is joy", [e["emotion"] for e in summary["most_common_emotions"]] == ["joy"])

print("\n2. Sentiment counts")
check(
    "counts are in the order Positive, Neutral, Negative: 3, 2, 2",
    [(i["sentiment"], i["count"]) for i in result["sentiment_counts"]] == [("Positive", 3), ("Neutral", 2), ("Negative", 2)],
)
check(
    "percentages are of the 7 counted entries",
    [i["percent"] for i in result["sentiment_counts"]] == [round(100 * 3 / 7, 1), round(100 * 2 / 7, 1), round(100 * 2 / 7, 1)],
)

print("\n3. Emotion counts")
emotions = result["emotion_counts"]
check(
    "counts match the saved emotions (the broken record's emotion is left out)",
    {e["emotion"]: e["count"] for e in emotions} == {"joy": 2, "sadness": 1, "neutral": 1, "excitement": 1, "no_clear_emotion": 1},
)
check(
    "the most frequent emotion comes first, with its name, emoji, and percentage",
    emotions[0]["emotion"] == "joy" and emotions[0]["name"] == "Happiness"
    and emotions[0]["emoji"] == "😊" and emotions[0]["percent"] == round(100 * 2 / 6, 1),
)
check(
    "neutral and 'no clear emotion' are not counted as detected emotions",
    {e["emotion"]: e["is_emotion"] for e in emotions}
    == {"joy": True, "sadness": True, "excitement": True, "neutral": False, "no_clear_emotion": False},
)
check("the counts add up to the entries that have an emotion", sum(e["count"] for e in emotions) == summary["entries_with_emotion"])

print("\n4. Emotional rhythm (line chart)")
rhythm = result["rhythm"]
check("one point per entry, numbered 1 to 7, oldest first", [p["number"] for p in rhythm] == list(range(1, 8)))
check("the points hold the stored compound scores, in order", [p["compound_score"] for p in rhythm] == scores)
times = [p["created_at"] for p in rhythm]
check("the points are in chronological order", times == sorted(times))
check("emotion names appear where saved, and not on the old entry", rhythm[0]["emotion_name"] == "Happiness" and rhythm[4]["emotion_name"] is None)
newest = dashboard_for(alice_id, rhythm_limit=3)["rhythm"]
check(
    "a limit of 3 gives the 3 newest entries with their real numbers",
    [p["number"] for p in newest] == [5, 6, 7] and [p["compound_score"] for p in newest] == scores[-3:],
)
check("long text is shortened and extra spaces are removed", preview("x" * 200) == "x" * 77 + "..." and preview("a   b\n c") == "a b c")

print("\n5. Edge cases")
carol_id, carol = register("carol@example.com")
empty = dashboard_for(carol_id)
check(
    "a user with no entries gets empty, honest numbers",
    empty["summary"]["total_entries"] == 0 and empty["summary"]["average_compound"] is None
    and empty["rhythm"] == [] and empty["emotion_counts"] == [] and empty["summary"]["most_common_sentiments"] == []
    and all(i["count"] == 0 and i["percent"] == 0.0 for i in empty["sentiment_counts"]),
)
dan_id, dan = register("dan@example.com")
add_rows(dan_id, [("a", 0.5, "Positive", None), ("b", 0.6, "Positive", None), ("c", -0.5, "Negative", None), ("d", -0.6, "Negative", None)])
check("a tie lists every tied sentiment", dashboard_for(dan_id)["summary"]["most_common_sentiments"] == ["Negative", "Positive"])
bob_result = dashboard_for(bob_id)
check(
    "another user's entries are never included",
    bob_result["summary"]["total_entries"] == 1 and [p["text_preview"] for p in bob_result["rhythm"]] == ["Bob's private note"]
    and dashboard_for(alice_id)["summary"]["total_entries"] == 7,
)
check("top_labels handles empty input and ties", top_labels({}) == [] and top_labels({"b": 2, "a": 2, "c": 1}) == ["a", "b"])

print("\n6. The API endpoint")
check("without a login token the dashboard returns 401", client.get("/api/dashboard").status_code == 401)
response = client.get("/api/dashboard", headers=alice)
api = response.json()
check(
    "with a token it returns 200 and the same numbers",
    response.status_code == 200 and api["summary"]["total_entries"] == 7
    and api["sentiment_counts"] == result["sentiment_counts"]
    and [p["compound_score"] for p in api["rhythm"]] == scores,
)
check("rhythm_limit=0 is rejected", client.get("/api/dashboard?rhythm_limit=0", headers=alice).status_code == 422)
check("rhythm_limit=201 is rejected", client.get("/api/dashboard?rhythm_limit=201", headers=alice).status_code == 422)
with patch.object(main, "build_dashboard", side_effect=RuntimeError("secret detail")):
    response = client.get("/api/dashboard", headers=alice)
check("an unexpected error returns a safe 500", response.status_code == 500 and "secret" not in response.text)
check("each user gets their own dashboard", client.get("/api/dashboard", headers=bob).json()["summary"]["total_entries"] == 1)

env_patch.stop()
print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)