"""Checks the database tables using a temporary in-memory database.

Run from the project root:
    python -m backend.tests.check_database
"""

import sys

from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app import models
from backend.app.database import Base
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


engine = create_engine("sqlite:///:memory:")  # temporary, disappears afterwards
Base.metadata.create_all(engine)

print("1. Tables")
tables = set(inspect(engine).get_table_names())
check("'users' and 'analyses' tables exist", tables == {"users", "analyses"})
user_columns = {column["name"] for column in inspect(engine).get_columns("users")}
check("users has email and password_hash columns", {"email", "password_hash"} <= user_columns)

print("\n2. Users")
with Session(engine) as session:
    session.add(
        models.User(full_name="Test User", email="test@example.com", password_hash="not-a-real-hash")
    )
    session.commit()
    found = session.scalars(select(models.User).where(models.User.email == "test@example.com")).one()
    check("a saved user can be read back", found.full_name == "Test User")

    session.add(models.User(full_name="Other", email="test@example.com", password_hash="x"))
    try:
        session.commit()
        duplicate_rejected = False
    except IntegrityError:
        session.rollback()
        duplicate_rejected = True
    check("a duplicate email is rejected by the database", duplicate_rejected)

    print("\n3. Analyses (using a real result from the sentiment pipeline)")
    result = analyze_text("I am very happy today. Everything is going well!")
    session.add(models.Analysis(user_id=found.id, **result))
    session.commit()

    session.refresh(found)
    check("the analysis is linked to its user", len(found.analyses) == 1)
    saved = found.analyses[0]
    check("scores and label are stored exactly", (
        saved.compound_score == result["compound_score"]
        and saved.sentiment == result["sentiment"]
        and saved.original_text == result["original_text"]
    ))

    print("\n4. Deleting a user")
    session.delete(found)
    session.commit()
    remaining = session.scalar(select(func.count()).select_from(models.Analysis))
    check("deleting a user also deletes their analyses", remaining == 0)

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)