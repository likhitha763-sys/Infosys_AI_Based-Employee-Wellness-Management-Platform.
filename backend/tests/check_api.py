"""End-to-end checks of the API (accounts, login protection, saved history).

Uses a temporary in-memory database. Run from the project root:
    python -m backend.tests.check_api
"""

import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app import models  # noqa: F401  (registers the tables)
from backend.app.database import Base, get_db
from backend.app.main import app

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
engine = create_engine(
    "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
)
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

# A temporary secret key, so the test does not depend on your real one.
key_patch = patch.dict(os.environ, {"SECRET_KEY": "t" * 64, "EMOTION_ENABLED": "false"})
key_patch.start()


def register_and_login(email):
    """Create an account and return the headers that prove the login."""
    body = {"full_name": "Test", "email": email, "password": "GoodPass123"}
    client.post("/api/auth/register", json=body)
    response = client.post(
        "/api/auth/login", json={"email": email, "password": "GoodPass123"}
    )
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def expected_label(compound):
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


print("1. Accounts")
check("health check works without login", client.get("/health").status_code == 200)
body = {"full_name": "Alice", "email": "alice@example.com", "password": "GoodPass123"}
registered = client.post("/api/auth/register", json=body)
check("registration returns 201", registered.status_code == 201)
check("the password is never returned", "password" not in registered.text)
check("a duplicate email returns 409", client.post("/api/auth/register", json=body).status_code == 409)
weak = {**body, "email": "weak@example.com", "password": "abc"}
check("a weak password returns 400", client.post("/api/auth/register", json=weak).status_code == 400)
wrong = client.post("/api/auth/login", json={"email": "alice@example.com", "password": "WrongPass123"})
unknown = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "GoodPass123"})
check("a wrong password returns 401", wrong.status_code == 401)
check("an unknown email gives the same 401 message", unknown.status_code == 401 and unknown.json() == wrong.json())

print("\n2. Analysis needs a login")
check("analyze without a token returns 401", client.post("/api/analyze", json={"text": "I am happy"}).status_code == 401)
check("batch without a token returns 401", client.post("/api/analyze/batch", json={"texts": ["I am happy"]}).status_code == 401)
check("history without a token returns 401", client.get("/api/history").status_code == 401)
bad_token = {"Authorization": "Bearer not.a.token"}
check("a fake token returns 401", client.post("/api/analyze", json={"text": "I am happy"}, headers=bad_token).status_code == 401)

print("\n3. Analyze and save")
alice = register_and_login("alice@example.com")
first_text = "I am very happy today. Everything is going well!"
response = client.post("/api/analyze", json={"text": first_text}, headers=alice)
check("analyze with a token returns 200", response.status_code == 200)
result = response.json()
check("the label follows the VADER thresholds", result["sentiment"] == expected_label(result["compound_score"]))
check("empty text returns 400", client.post("/api/analyze", json={"text": ""}, headers=alice).status_code == 400)
check("whitespace-only text returns 400", client.post("/api/analyze", json={"text": "   "}, headers=alice).status_code == 400)
history = client.get("/api/history", headers=alice).json()
check("exactly one result was saved (failed requests save nothing)", history["total"] == 1)
check("the saved text matches", history["items"][0]["original_text"] == first_text)
check("the saved score matches the response", history["items"][0]["compound_score"] == result["compound_score"])
check("the saved time includes a time zone", history["items"][0]["created_at"].endswith(("Z", "+00:00")))

print("\n4. Batch and ordering")
texts = ["I feel very sad and disappointed today.", "The meeting starts at 10 AM tomorrow.", "I am not happy with the result."]
batch = client.post("/api/analyze/batch", json={"texts": texts}, headers=alice)
check("batch returns 200 with 3 results", batch.status_code == 200 and batch.json()["count"] == 3)
check("a batch with a blank entry returns 400", client.post("/api/analyze/batch", json={"texts": ["fine text", "  "]}, headers=alice).status_code == 400)
history = client.get("/api/history", headers=alice).json()
check("history now has 4 results", history["total"] == 4)
check("the newest result comes first", history["items"][0]["original_text"] == texts[-1])
check("the oldest result comes last", history["items"][-1]["original_text"] == first_text)

print("\n5. Paging")
page = client.get("/api/history?limit=2", headers=alice).json()
check("limit=2 returns 2 items but total stays 4", len(page["items"]) == 2 and page["total"] == 4)
page = client.get("/api/history?limit=2&offset=3", headers=alice).json()
check("offset=3 returns the last item", len(page["items"]) == 1)
check("limit=0 is rejected", client.get("/api/history?limit=0", headers=alice).status_code == 422)
check("limit=1000 is rejected", client.get("/api/history?limit=1000", headers=alice).status_code == 422)

print("\n6. Users cannot see each other's history")
bob = register_and_login("bob@example.com")
check("a new user starts with an empty history", client.get("/api/history", headers=bob).json()["total"] == 0)
client.post("/api/analyze", json={"text": "Bob wrote this secret note."}, headers=bob)
bob_history = client.get("/api/history", headers=bob).json()
alice_history = client.get("/api/history", headers=alice).json()
check("Bob sees only his own result", bob_history["total"] == 1)
check("Alice's history did not change", alice_history["total"] == 4)
check("Bob's text never appears in Alice's history", "secret note" not in str(alice_history))

key_patch.stop()
print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)