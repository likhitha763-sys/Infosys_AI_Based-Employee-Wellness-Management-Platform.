"""End-to-end checks of emotion detection in the API, using fake emotion scores.

Uses a temporary in-memory database. Run from the project root:
    python -m backend.tests.check_emotion_api
"""

import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app import emotion, models
from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.sentiment import analyze_text

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


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

calls = []  # every list of texts the fake emotion model received


def fake_scores(texts):
    """Pretend model: texts containing 'happy' are joy, the rest sadness."""
    calls.append(list(texts))
    return [
        {"joy": 0.9, "excitement": 0.3, "neutral": 0.05, "sadness": 0.02}
        if "happy" in text.lower()
        else {"sadness": 0.8, "neutral": 0.1, "anger": 0.05, "joy": 0.02}
        for text in texts
    ]


env_patch = patch.dict(os.environ, {"SECRET_KEY": "t" * 64, "EMOTION_ENABLED": "true"})
model_patch = patch.object(emotion, "_predict_probabilities", side_effect=fake_scores)
env_patch.start()
model_patch.start()

body = {"full_name": "Alice", "email": "alice@example.com", "password": "GoodPass123"}
client.post("/api/auth/register", json=body)
token = client.post("/api/auth/login", json={"email": body["email"], "password": body["password"]}).json()["access_token"]
alice = {"Authorization": "Bearer " + token}
alice_id = client.get("/api/auth/me", headers=alice).json()["id"]

print("1. One text")
text = "I am very happy today. Everything is going well!"
response = client.post("/api/analyze", json={"text": text}, headers=alice)
result = response.json()
check("analyze returns 200", response.status_code == 200)
check(
    "emotion, name, emoji, and confidence come from the model scores",
    result["predicted_emotion"] == "joy" and result["emotion_name"] == "Happiness"
    and result["emotion_emoji"] == "😊" and result["emotion_confidence"] == 0.9,
)
check(
    "the top 3 emotions are listed, best first",
    [item["emotion"] for item in result["top_emotions"]] == ["joy", "excitement", "neutral"],
)
check("there is no note when emotion detection works", result["emotion_note"] is None)
direct = analyze_text(text)
check(
    "the sentiment result is the same as without emotion detection",
    result["sentiment"] == direct["sentiment"] and result["compound_score"] == direct["compound_score"],
)
check("the emotion model received the original text", calls[-1] == [text])

print("\n2. Saved history")
item = client.get("/api/history", headers=alice).json()["items"][0]
check(
    "history shows the saved emotion with its name and emoji",
    item["predicted_emotion"] == "joy" and item["emotion_name"] == "Happiness" and item["emotion_emoji"] == "😊",
)
check("the saved confidence matches", item["emotion_confidence"] == 0.9)
check("history does not include the top-3 list (it is not stored)", item["top_emotions"] is None)

print("\n3. Several texts")
texts = ["I am happy with my results.", "This is a sad and lonely evening."]
before = len(calls)
batch = client.post("/api/analyze/batch", json={"texts": texts}, headers=alice)
check("batch returns 200 with 2 results", batch.status_code == 200 and batch.json()["count"] == 2)
check("each entry has its own emotion, in order", [r["predicted_emotion"] for r in batch.json()["results"]] == ["joy", "sadness"])
check("the model is called once for the whole batch", len(calls) == before + 1 and calls[-1] == texts)
history = client.get("/api/history", headers=alice).json()
check("history has 3 results with the newest first", history["total"] == 3 and history["items"][0]["original_text"] == texts[1])

print("\n4. Results saved before emotions existed")
with TestingSession() as session:
    session.add(models.Analysis(
        user_id=alice_id, original_text="Old entry from before emotions", processed_text="old entry",
        positive_score=0.0, negative_score=0.0, neutral_score=1.0, compound_score=0.0, sentiment="Neutral",
    ))
    session.commit()
history = client.get("/api/history", headers=alice)
old_items = [i for i in history.json()["items"] if i["original_text"].startswith("Old entry")]
check(
    "an old result appears in history with empty emotion fields",
    history.status_code == 200 and len(old_items) == 1
    and old_items[0]["predicted_emotion"] is None and old_items[0]["emotion_name"] is None,
)

print("\n5. Emotion detection off or failing")
with patch.dict(os.environ, {"EMOTION_ENABLED": "false"}):
    response = client.post("/api/analyze", json={"text": "I am happy today"}, headers=alice)
    off = response.json()
check("switched off: the analysis still returns 200 with a sentiment", response.status_code == 200 and off["sentiment"] in ("Positive", "Negative", "Neutral"))
check("switched off: emotion fields are empty and a note explains why", off["predicted_emotion"] is None and "turned off" in off["emotion_note"])
newest = client.get("/api/history", headers=alice).json()["items"][0]
check("switched off: the result is saved with no emotion", newest["original_text"] == "I am happy today" and newest["predicted_emotion"] is None)

with patch.object(emotion, "_predict_probabilities", side_effect=emotion.EmotionModelError("secret model path")):
    response = client.post("/api/analyze", json={"text": "I am happy today"}, headers=alice)
check(
    "model missing: still 200, a short note, and no internal details",
    response.status_code == 200 and response.json()["predicted_emotion"] is None
    and "unavailable" in response.json()["emotion_note"] and "secret" not in response.text,
)
with patch.object(emotion, "_predict_probabilities", side_effect=RuntimeError("secret internal detail")):
    response = client.post("/api/analyze", json={"text": "I am happy today"}, headers=alice)
check(
    "unexpected error: still 200, a note, and no internal details",
    response.status_code == 200 and response.json()["emotion_note"] and "secret" not in response.text,
)
with patch.object(emotion, "_predict_probabilities", side_effect=emotion.EmotionModelError("x")):
    response = client.post("/api/analyze/batch", json={"texts": texts}, headers=alice)
check(
    "batch with emotion unavailable: every entry still has a sentiment",
    response.status_code == 200 and all(r["predicted_emotion"] is None and r["sentiment"] for r in response.json()["results"]),
)

print("\n6. Existing checks still hold")
before = len(calls)
check("empty text still returns 400", client.post("/api/analyze", json={"text": "  "}, headers=alice).status_code == 400)
check("a blank batch entry still returns 400", client.post("/api/analyze/batch", json={"texts": ["fine", " "]}, headers=alice).status_code == 400)
check("blank input never reaches the emotion model", len(calls) == before)
check("analyze without a login still returns 401", client.post("/api/analyze", json={"text": "I am happy"}).status_code == 401)

model_patch.stop()
env_patch.stop()
print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)