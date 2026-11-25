"""Checks for frontend/goal_helpers.py.

Run from the project root:
    python -m frontend.tests.check_goal_helpers
"""

import sys
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")

from frontend.goal_helpers import (  # noqa: E402
    escape_markdown, goal_due_text, parse_date, progress_counts,
)

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


today = date(2026, 11, 25)

check("a valid date is parsed", parse_date("2026-12-31") == date(2026, 12, 31))
check("an empty value gives None", parse_date(None) is None and parse_date("") is None)
check("a bad value gives None", parse_date("not-a-date") is None)

check("no target date gives an empty note", goal_due_text({"completed": False}, today) == "")
check("a past open goal is overdue",
      goal_due_text({"target_date": "2026-11-20", "completed": False}, today) == "Overdue since 2026-11-20")
check("today is due today", goal_due_text({"target_date": "2026-11-25", "completed": False}, today) == "Due today")
check("tomorrow is due tomorrow",
      goal_due_text({"target_date": "2026-11-26", "completed": False}, today) == "Due tomorrow")
check("later shows the date",
      goal_due_text({"target_date": "2026-12-31", "completed": False}, today) == "Due 2026-12-31")
check("a completed goal is never overdue",
      "Overdue" not in goal_due_text({"target_date": "2026-11-20", "completed": True}, today))

check("progress counts are correct",
      progress_counts([{"completed": True}, {"completed": False}, {"completed": True}]) == (2, 3))
check("progress counts for no goals", progress_counts([]) == (0, 0))

check("markdown characters are escaped", escape_markdown("a*b_c") == "a\\*b\\_c")
check("plain text is unchanged", escape_markdown("Walk 10 minutes") == "Walk 10 minutes")

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)