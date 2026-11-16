"""Checks save=false, the data export, and deleting saved analyses.

Run from the project root:
    python -m backend.tests.check_privacy
It creates two throwaway accounts with random emails.
"""

import os
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
os.environ["EMOTION_ENABLED"] = "false"  # not needed here, and faster

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
    email = f"privacy_{tag}_{uuid.uuid4().hex[:8]}@example.com"
    body = {"full_name": f"Privacy {tag}", "email": email, "password": "GoodPass123"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    login = client.post("/api/auth/login", json={"email": email, "password": "GoodPass123"})
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def total(client, headers):
    return client.get("/api/history", params={"limit": 100}, headers=headers).json()["total"]


with TestClient(app) as client:
    a = make_user(client, "a")
    b = make_user(client, "b")
    one = {"text": "I am very happy today."}
    two = {"texts": ["I feel sad today.", "This is fine."]}

    r = client.post("/api/analyze", params={"save": "false"}, json=one, headers=a)
    check("analyze with save=false returns 200", r.status_code == 200)
    check("nothing was saved after save=false", total(client, a) == 0)

    client.post("/api/analyze", json=one, headers=a)
    check("analyze without the option saves the result", total(client, a) == 1)

    client.post("/api/analyze/batch", params={"save": "false"}, json=two, headers=a)
    check("batch with save=false saves nothing", total(client, a) == 1)
    client.post("/api/analyze/batch", json=two, headers=a)
    check("batch without the option saves both results", total(client, a) == 3)

    client.post("/api/analyze", json=one, headers=b)
    check("the other user has their own single entry", total(client, b) == 1)

    export = client.get("/api/history/export", headers=a).json()
    check("export returns all 3 entries", export["total"] == 3 and len(export["items"]) == 3)
    check("export contains the saved text",
          any(item["original_text"] == "I am very happy today." for item in export["items"]))
    check("export does not include the other user's data", all(
        item["original_text"] != "" for item in export["items"]) and len(export["items"]) == 3)

    check("export needs a login", client.get("/api/history/export").status_code == 401)
    check("delete needs a login", client.delete("/api/history").status_code == 401)

    deleted = client.delete("/api/history", headers=a)
    check("delete returns 200 and the number deleted", deleted.status_code == 200 and deleted.json()["deleted"] == 3)
    check("history is empty after delete", total(client, a) == 0)
    check("the other user's entry is untouched", total(client, b) == 1)
    check("deleting again deletes 0", client.delete("/api/history", headers=a).json()["deleted"] == 0)

    client.delete("/api/history", headers=b)  # clean up this test's data

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)