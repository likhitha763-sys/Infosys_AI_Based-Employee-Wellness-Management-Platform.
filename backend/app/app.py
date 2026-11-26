"""Saving and reading wellness goals (database logic).

This module contains no web code. FastAPI (main.py) calls these functions.
Every function filters by user_id, so one user can never reach another's goals.
"""

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .models import Goal, utc_now

MAX_TITLE_LENGTH = 200
MAX_GOALS_PER_USER = 50


class GoalNotFoundError(Exception):
    """The goal does not exist, or it belongs to another user."""


class GoalLimitError(Exception):
    """The user already has the maximum number of goals."""


def count_goals(db: Session, user_id):
    """How many goals a user has."""
    return db.scalar(select(func.count()).select_from(Goal).where(Goal.user_id == user_id))


def create_goal(db: Session, user_id, title, target_date=None):
    """Create a goal. Raises ValueError for a bad title and GoalLimitError at the limit."""
    title = (title or "").strip()
    if not title:
        raise ValueError("Goal title must not be empty.")
    if len(title) > MAX_TITLE_LENGTH:
        raise ValueError(f"Goal title must be at most {MAX_TITLE_LENGTH} characters.")
    if count_goals(db, user_id) >= MAX_GOALS_PER_USER:
        raise GoalLimitError(
            f"You can have at most {MAX_GOALS_PER_USER} goals. "
            "Please delete one before adding another."
        )
    goal = Goal(user_id=user_id, title=title, target_date=target_date)
    db.add(goal)
    db.commit()
    db.refresh(goal)
    return goal


def list_goals(db: Session, user_id):
    """A user's goals: open ones first, newest first within each group."""
    query = (
        select(Goal)
        .where(Goal.user_id == user_id)
        .order_by(Goal.completed, Goal.created_at.desc(), Goal.id.desc())
    )
    return list(db.scalars(query))


def _get_goal(db: Session, user_id, goal_id):
    goal = db.scalar(select(Goal).where(Goal.id == goal_id, Goal.user_id == user_id))
    if goal is None:
        raise GoalNotFoundError()
    return goal


def set_goal_completed(db: Session, user_id, goal_id, completed):
    """Mark a goal done or reopen it."""
    goal = _get_goal(db, user_id, goal_id)
    goal.completed = bool(completed)
    goal.completed_at = utc_now() if completed else None
    db.commit()
    db.refresh(goal)
    return goal


def delete_goal(db: Session, user_id, goal_id):
    """Delete one goal."""
    goal = _get_goal(db, user_id, goal_id)
    db.delete(goal)
    db.commit()


def delete_all_goals(db: Session, user_id):
    """Delete every goal of a user. Returns how many were deleted."""
    result = db.execute(delete(Goal).where(Goal.user_id == user_id))
    db.commit()
    return result.rowcount