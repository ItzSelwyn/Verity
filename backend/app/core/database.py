"""
database.py — Database engine, session, and Base setup for Verity API.

We use SQLAlchemy with SQLite for local development. The `get_db` function is
a FastAPI dependency that provides a database session per request and ensures
the session is always closed after the request completes.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import DATABASE_URL

# -----------------------------------------------------------------
# Engine
# -----------------------------------------------------------------
# `check_same_thread=False` is required for SQLite when used with FastAPI
# because FastAPI can serve requests from multiple threads.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# -----------------------------------------------------------------
# Session factory
# -----------------------------------------------------------------
# `autocommit=False` means we control transactions manually.
# `autoflush=False` means changes are only sent to the DB when we commit.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# -----------------------------------------------------------------
# Declarative Base
# -----------------------------------------------------------------
# All ORM models inherit from this Base so SQLAlchemy knows about them.
Base = declarative_base()


# -----------------------------------------------------------------
# FastAPI dependency
# -----------------------------------------------------------------
def get_db():
    """
    Provide a database session for a single request.

    Usage in a route:
        def my_route(db: Session = Depends(get_db)):
            ...

    The session is automatically closed when the request finishes,
    even if an exception is raised.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
