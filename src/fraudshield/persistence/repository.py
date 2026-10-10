"""Repository for durable fraud decision history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fraudshield.persistence.models import FraudDecisionRecord


@dataclass(frozen=True, slots=True)
class FraudDecisionWrite:
    """Data required to persist one fraud decision."""

    transaction_id: int
    event_time: datetime
    customer_id: int
    terminal_id: int
    amount: float
    raw_probability: float
    calibrated_probability: float
    decision: str
    model_version: str
    policy_version: str


class DuplicateDecisionError(ValueError):
    """Raised when a transaction decision already exists."""


class FraudDecisionRepository:
    """Persistence operations for fraud decisions."""

    def __init__(
        self,
        session: Session,
    ) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        """Expose session for read-only aggregate queries."""
        return self._session

    def add(
        self,
        decision: FraudDecisionWrite,
    ) -> FraudDecisionRecord:
        """Persist one fraud decision."""
        record = FraudDecisionRecord(
            transaction_id=decision.transaction_id,
            event_time=decision.event_time,
            customer_id=decision.customer_id,
            terminal_id=decision.terminal_id,
            amount=decision.amount,
            raw_probability=decision.raw_probability,
            calibrated_probability=(
                decision.calibrated_probability
            ),
            decision=decision.decision,
            model_version=decision.model_version,
            policy_version=decision.policy_version,
        )

        self._session.add(record)

        try:
            self._session.flush()
        except IntegrityError as exc:
            self._session.rollback()

            raise DuplicateDecisionError(
                f"Decision already exists for transaction "
                f"{decision.transaction_id}."
            ) from exc

        return record

    def get_by_transaction_id(
        self,
        transaction_id: int,
    ) -> FraudDecisionRecord | None:
        """Return one decision by transaction ID."""
        statement = select(
            FraudDecisionRecord
        ).where(
            FraudDecisionRecord.transaction_id
            == transaction_id
        )

        return self._session.scalar(statement)

    def recent(
        self,
        limit: int = 100,
    ) -> list[FraudDecisionRecord]:
        """Return most recent fraud decisions."""
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        statement = (
            select(FraudDecisionRecord)
            .order_by(
                FraudDecisionRecord.event_time.desc(),
                FraudDecisionRecord.transaction_id.desc(),
            )
            .limit(limit)
        )

        return list(
            self._session.scalars(statement)
        )

    def count(self) -> int:
        """Return total number of persisted decisions."""
        statement = select(
            func.count(FraudDecisionRecord.id)
        )

        result = self._session.scalar(statement)

        return int(result or 0)