"""Checks the Wellness Resources endpoint.

Run from the project root:
    python -m backend.tests.check_resources
It creates one throwaway account with a random email.
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


with TestClient(app) as client:
    email = f"resources_{uuid.uuid4().hex[:8]}@example.com"
    body = {"full_name": "Resources Test", "email": email, "password": "GoodPass123"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    login = client.post("/api/auth/login", json={"email": email, "password": "GoodPass123"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    check("resources need a login", client.get("/api/resources").status_code == 401)

    response = client.get("/api/resources", headers=headers)
    check("resources return 200 with a login", response.status_code == 200)
    data = response.json()

    check("a disclaimer is included", bool(data["disclaimer"].strip()))
    check("the disclaimer says this is not medical advice", "not medical advice" in data["disclaimer"].lower())
    check("there are at least 5 guides", len(data["items"]) >= 5)
    ids = [item["id"] for item in data["items"]]
    check("guide ids are unique", len(ids) == len(set(ids)))
    check("every guide has a title, a summary and steps",
          all(item["title"] and item["summary"] and item["steps"] for item in data["items"]))
    contacts = " ".join(c["contact"] for c in data["help_contacts"])
    check("the emergency number 112 is listed", "112" in contacts)
    check("Tele-MANAS 14416 is listed", "14416" in contacts)

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)