"""Checks that old databases get the new emotion columns without losing data.

Uses temporary database files. Run from the project root:
    python -m backend.tests.check_migration
"""

import sys
import tempfile
from pathlib import Path

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from backend.app import models
from backend.app.database import Base, add_missing_columns

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


def column_names(engine, table):
    return {column["name"] for column in inspect(engine).get_columns(table)}


# The analyses table exactly as it was before emotions were added.
OLD_ANALYSES_TABLE = """
CREATE TABLE analyses (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    original_text TEXT NOT NULL,
    processed_text TEXT NOT NULL,
    positive_score FLOAT NOT NULL,
    negative_score FLOAT NOT NULL,
    neutral_score FLOAT NOT NULL,
    compound_score FLOAT NOT NULL,
    sentiment VARCHAR(10) NOT NULL,
    created_at DATETIME NOT NULL
)
"""

engines = []
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    print("1. An old database with a saved result")
    old = create_engine(f"sqlite:///{Path(folder) / 'old.db'}")
    engines.append(old)
    models.User.__table__.create(old)
    with old.begin() as connection:
        connection.execute(text(OLD_ANALYSES_TABLE))
        connection.execute(text(
            "INSERT INTO users (id, full_name, email, password_hash, created_at) "
            "VALUES (1, 'Old User', 'old@example.com', 'x', '2026-10-01 10:00:00')"
        ))
        connection.execute(text(
            "INSERT INTO analyses (id, user_id, original_text, processed_text, positive_score, "
            "negative_score, neutral_score, compound_score, sentiment, created_at) "
            "VALUES (1, 1, 'old text', 'old text', 0.5, 0.0, 0.5, 0.6, 'Positive', '2026-10-01 10:00:00')"
        ))
    check("the old table has no emotion columns", "predicted_emotion" not in column_names(old, "analyses"))

    added = add_missing_columns(old)
    check("the update adds exactly the two emotion columns", sorted(added) == ["emotion_confidence", "predicted_emotion"])
    check("the columns now exist", {"predicted_emotion", "emotion_confidence"} <= column_names(old, "analyses"))

    with Session(old) as session:
        row = session.scalars(select(models.Analysis)).one()
        check(
            "the old saved result is kept, with empty emotion fields",
            row.original_text == "old text" and row.compound_score == 0.6
            and row.predicted_emotion is None and row.emotion_confidence is None,
        )
        check("running the update again changes nothing", add_missing_columns(old) == [])

        session.add(models.Analysis(
            user_id=1, original_text="new", processed_text="new", positive_score=0.5,
            negative_score=0.0, neutral_score=0.5, compound_score=0.5, sentiment="Positive",
            predicted_emotion="joy", emotion_confidence=0.9,
        ))
        session.commit()
        saved = session.scalars(select(models.Analysis).where(models.Analysis.original_text == "new")).one()
        check("a new result with an emotion can be saved and read", saved.predicted_emotion == "joy" and saved.emotion_confidence == 0.9)

    print("\n2. A brand-new database")
    fresh = create_engine(f"sqlite:///{Path(folder) / 'fresh.db'}")
    engines.append(fresh)
    Base.metadata.create_all(fresh)
    check("new tables already include the emotion columns", {"predicted_emotion", "emotion_confidence"} <= column_names(fresh, "analyses"))
    check("the update has nothing to add", add_missing_columns(fresh) == [])

    print("\n3. An empty database")
    empty = create_engine(f"sqlite:///{Path(folder) / 'empty.db'}")
    engines.append(empty)
    check("a database with no tables does not cause an error", add_missing_columns(empty) == [])

    for engine in engines:
        engine.dispose()

print("\n" + "=" * 50)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 50)
sys.exit(1 if failed else 0)