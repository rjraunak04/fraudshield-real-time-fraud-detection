"""SQLAlchemy persistence models for FraudShield."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from fraudshield.persistence.database import Base


class FraudDecisionRecord(Base):
    """Durable audit record for one fraud-scoring decision."""

    __tablename__ = "fraud_decisions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    transaction_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
        index=True,
    )

    event_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        index=True,
    )

    customer_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    terminal_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    amount: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    raw_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    calibrated_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
    )

    decision: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        index=True,
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    policy_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )