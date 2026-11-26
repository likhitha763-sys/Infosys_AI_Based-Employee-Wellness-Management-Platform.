"""Small text helpers for the Wellness Goals page (no screen code)."""

from datetime import date

_MARKDOWN_SPECIAL = "\\`*_{}[]()#+!|<>~"


def parse_date(value):
    """An ISO date string as a date, or None if it is empty or invalid."""
    try:
        return date.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def goal_due_text(goal, today=None):
    """A short note about a goal's target date ('' when it has none)."""
    target = parse_date(goal.get("target_date"))
    if target is None:
        return ""
    if goal.get("completed"):
        return f"Target date {target.isoformat()}"
    days = (target - (today or date.today())).days
    if days < 0:
        return f"Overdue since {target.isoformat()}"
    if days == 0:
        return "Due today"
    if days == 1:
        return "Due tomorrow"
    return f"Due {target.isoformat()}"


def progress_counts(goals):
    """(completed, total) for a list of goals."""
    return sum(1 for goal in goals if goal.get("completed")), len(goals)


def escape_markdown(text):
    """Make user text safe to show with st.markdown (no formatting surprises)."""
    return "".join("\\" + ch if ch in _MARKDOWN_SPECIAL else ch for ch in str(text))