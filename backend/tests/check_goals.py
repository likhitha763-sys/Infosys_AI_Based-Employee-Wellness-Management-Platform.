"""Checks the wellness goals endpoints.

Run from the project root:
    python -m backend.tests.check_goals
It creates two throwaway accounts with random emails.
"""

import os
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
os.environ["EMOTION_ENABLED"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

passed = 0
failed = 0


def check(description, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {description}")
    else:
        failed += 1
        print(f"FAIL  {description}")


def make_user(client, tag):
    email = f"goals_{tag}_{uuid.uuid4().hex[:8]}@example.com"
    body = {"full_name": f"Goals {tag}", "email": email, "password": "GoodPass123"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    login = client.post("/api/auth/login", json={"email": email, "password": "GoodPass123"})
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


with TestClient(app) as client:
    a = make_user(client, "a")
    b = make_user(client, "b")

    check("listing goals needs a login", client.get("/api/goals").status_code == 401)
    check("adding a goal needs a login", client.post("/api/goals", json={"title": "x"}).status_code == 401)
    check("a new user has no goals", client.get("/api/goals", headers=a).json()["total"] == 0)

    created = client.post("/api/goals", json={"title": "  Walk 10 minutes  ", "target_date": "2026-12-31"}, headers=a)
    check("adding a goal returns 201", created.status_code == 201)
    goal = created.json()
    check("the title is trimmed", goal["title"] == "Walk 10 minutes")
    check("the target date is kept", goal["target_date"] == "2026-12-31")
    check("a new goal is not completed", goal["completed"] is False and goal["completed_at"] is None)

    no_date = client.post("/api/goals", json={"title": "Sleep earlier"}, headers=a)
    check("a goal without a date is allowed", no_date.status_code == 201 and no_date.json()["target_date"] is None)

    check("a blank title is rejected", client.post("/api/goals", json={"title": "   "}, headers=a).status_code == 422)
    check("a title over 200 characters is rejected",
          client.post("/api/goals", json={"title": "x" * 201}, headers=a).status_code == 422)
    check("a bad date is rejected",
          client.post("/api/goals", json={"title": "x", "target_date": "not-a-date"}, headers=a).status_code == 422)
    check("a missing title is rejected", client.post("/api/goals", json={}, headers=a).status_code == 422)

    done = client.patch(f"/api/goals/{goal['id']}", json={"completed": True}, headers=a).json()
    check("completing a goal sets completed and a time", done["completed"] is True and done["completed_at"] is not None)

    listing = client.get("/api/goals", headers=a).json()
    check("two goals are listed", listing["total"] == 2)
    check("open goals are listed before completed ones", listing["items"][0]["completed"] is False)

    reopened = client.patch(f"/api/goals/{goal['id']}", json={"completed": False}, headers=a).json()
    check("reopening clears completed and the time", reopened["completed"] is False and reopened["completed_at"] is None)

    check("the other user sees no goals", client.get("/api/goals", headers=b).json()["total"] == 0)
    check("the other user cannot update my goal",
          client.patch(f"/api/goals/{goal['id']}", json={"completed": True}, headers=b).status_code == 404)
    check("the other user cannot delete my goal",
          client.delete(f"/api/goals/{goal['id']}", headers=b).status_code == 404)
    check("my goal is still there after those attempts", client.get("/api/goals", headers=a).json()["total"] == 2)
    check("updating a missing goal returns 404",
          client.patch("/api/goals/99999999", json={"completed": True}, headers=a).status_code == 404)

    removed = client.delete(f"/api/goals/{goal['id']}", headers=a)
    check("deleting my goal works", removed.status_code == 200 and removed.json()["deleted"] == 1)
    check("one goal is left", client.get("/api/goals", headers=a).json()["total"] == 1)

    for number in range(49):
        client.post("/api/goals", json={"title": f"Goal {number}"}, headers=a)
    check("a user can have 50 goals", client.get("/api/goals", headers=a).json()["total"] == 50)
    over = client.post("/api/goals", json={"title": "One too many"}, headers=a)
    check("the 51st goal is refused with a clear message", over.status_code == 400 and "at most 50" in over.json()["detail"])

    wiped = client.delete("/api/goals", headers=a)
    check("deleting all goals returns how many", wiped.status_code == 200 and wiped.json()["deleted"] == 50)
    check("no goals are left", client.get("/api/goals", headers=a).json()["total"] == 0)

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)