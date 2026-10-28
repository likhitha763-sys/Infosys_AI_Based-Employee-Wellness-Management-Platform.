"""Database connection (SQLAlchemy).

Uses SQLite by default. To use MySQL later, only DATABASE_URL in
backend/.env needs to change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Read settings from backend/.env if the file exists.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# For SQLite this file is created in the folder you run the app from
# (the project root).
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./moodmentor.db")

# SQLite needs this option because FastAPI uses several threads.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Columns added after a table was first created: {table: {column: SQL type}}.
# create_all() does not change tables that already exist, so these are added
# separately. Existing rows are kept, and the new columns are empty for them.
LATER_COLUMNS = {
    "analyses": {
        "predicted_emotion": "VARCHAR(30)",
        "emotion_confidence": "FLOAT",
    },
}


class Base(DeclarativeBase):
    """Parent class of all database tables."""


def get_db():
    """FastAPI dependency: one session per request, always closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_missing_columns(db_engine=None):
    """Add any LATER_COLUMNS that are missing. Safe to run many times.

    Returns the list of columns that were added.
    """
    db_engine = db_engine or engine
    inspector = inspect(db_engine)
    existing_tables = set(inspector.get_table_names())
    added = []

    with db_engine.begin() as connection:
        for table, columns in LATER_COLUMNS.items():
            if table not in existing_tables:
                continue
            present = {column["name"] for column in inspector.get_columns(table)}
            for name, sql_type in columns.items():
                if name not in present:
                    # Names and types come from LATER_COLUMNS above, never from users.
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
                    added.append(name)
    return added


def create_tables():
    """Create any missing tables, then add any missing columns.

    Returns the list of columns that were added to existing tables.
    """
    from . import models  # noqa: F401  (importing registers the tables)

    Base.metadata.create_all(bind=engine)
    return add_missing_columns()