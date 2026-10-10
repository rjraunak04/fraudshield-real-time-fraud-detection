"""Database infrastructure for FraudShield persistence."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for FraudShield persistence models."""


def create_database_engine(
    database_url: str,
    *,
    echo: bool = False,
) -> Engine:
    """Create the SQLAlchemy database engine."""
    connect_args: dict[str, object] = {}

    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(
        database_url,
        echo=echo,
        future=True,
        connect_args=connect_args,
    )


def create_session_factory(
    engine: Engine,
) -> sessionmaker[Session]:
    """Create a database session factory."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )


def create_schema(engine: Engine) -> None:
    """Create all FraudShield persistence tables."""
    Base.metadata.create_all(engine)


def session_scope(
    session_factory: sessionmaker[Session],
) -> Generator[Session, None, None]:
    """Provide a transactional database session."""
    session = session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()