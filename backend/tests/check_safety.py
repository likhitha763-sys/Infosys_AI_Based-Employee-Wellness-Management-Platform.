"""Checks the urgent-text phrase check. Run: python -m backend.tests.check_safety"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from backend.app.safety import check_urgent  # noqa: E402

passed = failed = 0


def check(description, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {description}")
    else:
        failed += 1
        print(f"FAIL  {description}")


flagged = [
    "I want to kill myself",
    "I don\u2019t want to live anymore",
    "Sometimes I think about ending my life.",
    "I've been thinking about suicide",
    "I keep hurting myself",
    "Everyone would be better off dead without me",
]
not_flagged = [
    "I am very happy today. Everything is going well!",
    "This exam is killing me",
    "I'm excited about the trip but also really nervous.",
    "The movie was a killer",
    "ok",
    "",
]
for text in flagged:
    check(f"flagged: {text}", check_urgent(text))
for text in not_flagged:
    check(f"not flagged: {text!r}", not check_urgent(text))
check("a non-text value is not flagged and does not crash", check_urgent(None) is False)

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)