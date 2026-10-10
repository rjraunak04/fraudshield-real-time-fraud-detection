"""Tests for FraudShield durable decision persistence."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from fraudshield.persistence import (
    DuplicateDecisionError,
    FraudDecisionRepository,
    FraudDecisionWrite,
    create_database_engine,
    create_schema,
    create_session_factory,
)


@pytest.fixture
def session() -> Session:
    """Create an isolated in-memory persistence session."""
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)

    factory = create_session_factory(engine)
    database_session = factory()

    try:
        yield database_session
    finally:
        database_session.close()
        engine.dispose()


def make_decision(
    transaction_id: int = 1,
    *,
    event_time: datetime | None = None,
    decision: str = "REVIEW",
) -> FraudDecisionWrite:
    """Create a deterministic decision persistence payload."""
    return FraudDecisionWrite(
        transaction_id=transaction_id,
        event_time=event_time or datetime(2018, 4, 1, 12, 0, 0),
        customer_id=101,
        terminal_id=501,
        amount=125.50,
        raw_probability=0.061,
        calibrated_probability=0.060,
        decision=decision,
        model_version="day-03-hgb-balanced-v1",
        policy_version="day-04-three-way-policy-v1",
    )


def test_add_and_read_decision(
    session: Session,
) -> None:
    repository = FraudDecisionRepository(session)

    record = repository.add(make_decision())
    session.commit()

    stored = repository.get_by_transaction_id(1)

    assert record.id is not None
    assert stored is not None
    assert stored.transaction_id == 1
    assert stored.customer_id == 101
    assert stored.terminal_id == 501
    assert stored.amount == pytest.approx(125.50)
    assert stored.raw_probability == pytest.approx(0.061)
    assert stored.calibrated_probability == pytest.approx(0.060)
    assert stored.decision == "REVIEW"
    assert stored.model_version == "day-03-hgb-balanced-v1"
    assert stored.policy_version == "day-04-three-way-policy-v1"


def test_repository_count_starts_at_zero(
    session: Session,
) -> None:
    repository = FraudDecisionRepository(session)

    assert repository.count() == 0


def test_repository_count_tracks_records(
    session: Session,
) -> None:
    repository = FraudDecisionRepository(session)

    repository.add(make_decision(1))
    repository.add(make_decision(2))
    session.commit()

    assert repository.count() == 2


def test_missing_transaction_returns_none(
    session: Session,
) -> None:
    repository = FraudDecisionRepository(session)

    assert repository.get_by_transaction_id(999) is None


def test_duplicate_transaction_is_rejected(
    session: Session,
) -> None:
    repository = FraudDecisionRepository(session)

    repository.add(make_decision(1))
    session.commit()

    with pytest.raises(
        DuplicateDecisionError,
        match="transaction 1",
    ):
        repository.add(make_decision(1))


def test_session_remains_usable_after_duplicate(
    session: Session,
) -> None:
    repository = FraudDecisionRepository(session)

    repository.add(make_decision(1))
    session.commit()

    with pytest.raises(DuplicateDecisionError):
        repository.add(make_decision(1))

    repository.add(make_decision(2))
    session.commit()

    assert repository.count() == 2


def test_recent_orders_by_event_time_descending(
    session: Session,
) -> None:
    repository = FraudDecisionRepository(session)
    start = datetime(2018, 4, 1, 12, 0, 0)

    repository.add(
        make_decision(
            1,
            event_time=start,
        )
    )
    repository.add(
        make_decision(
            2,
            event_time=start + timedelta(minutes=2),
        )
    )
    repository.add(
        make_decision(
            3,
            event_time=start + timedelta(minutes=1),
        )
    )
    session.commit()

    records = repository.recent(limit=3)

    assert [
        record.transaction_id
        for record in records
    ] == [2, 3, 1]


def test_recent_uses_transaction_id_as_tie_breaker(
    session: Session,
) -> None:
    repository = FraudDecisionRepository(session)
    event_time = datetime(2018, 4, 1, 12, 0, 0)

    repository.add(
        make_decision(
            1,
            event_time=event_time,
        )
    )
    repository.add(
        make_decision(
            3,
            event_time=event_time,
        )
    )
    repository.add(
        make_decision(
            2,
            event_time=event_time,
        )
    )
    session.commit()

    records = repository.recent(limit=3)

    assert [
        record.transaction_id
        for record in records
    ] == [3, 2, 1]


def test_recent_respects_limit(
    session: Session,
) -> None:
    repository = FraudDecisionRepository(session)

    for transaction_id in range(1, 6):
        repository.add(
            make_decision(transaction_id)
        )

    session.commit()

    records = repository.recent(limit=2)

    assert len(records) == 2


@pytest.mark.parametrize(
    "limit",
    [0, -1],
)
def test_recent_rejects_invalid_limit(
    session: Session,
    limit: int,
) -> None:
    repository = FraudDecisionRepository(session)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        repository.recent(limit=limit)


@pytest.mark.parametrize(
    "decision",
    ["APPROVE", "REVIEW", "BLOCK"],
)
def test_three_way_decisions_can_be_persisted(
    session: Session,
    decision: str,
) -> None:
    repository = FraudDecisionRepository(session)

    record = repository.add(
        make_decision(
            transaction_id={
                "APPROVE": 1,
                "REVIEW": 2,
                "BLOCK": 3,
            }[decision],
            decision=decision,
        )
    )

    session.commit()

    assert record.decision == decision