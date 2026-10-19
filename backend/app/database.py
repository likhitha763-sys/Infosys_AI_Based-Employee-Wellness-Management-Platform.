"""Database connection (SQLAlchemy).

Uses SQLite by default. To use MySQL later, only DATABASE_URL in
backend/.env needs to change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
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


class Base(DeclarativeBase):
    """Parent class of all database tables."""


def get_db():
    """FastAPI dependency: one session per request, always closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create any tables that do not exist yet."""
    from . import models  # noqa: F401  (importing registers the tables)

    Base.metadata.create_all(bind=engine)