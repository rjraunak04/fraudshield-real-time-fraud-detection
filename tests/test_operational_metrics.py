"""Tests for FraudShield operational metrics."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine

from fraudshield.metrics import (
    build_operational_metrics,
)
from fraudshield.persistence import (
    FraudDecisionRepository,
    FraudDecisionWrite,
    create_schema,
    create_session_factory,
)


@pytest.fixture
def repository():
    """Create isolated decision repository."""
    engine = create_engine(
        "sqlite:///:memory:"
    )
    create_schema(engine)

    session_factory = create_session_factory(
        engine
    )
    session = session_factory()

    repo = FraudDecisionRepository(session)

    yield repo

    session.close()
    engine.dispose()


def add_decision(
    repository,
    *,
    transaction_id: int,
    amount: float,
    probability: float,
    decision: str,
) -> None:
    """Persist one deterministic decision."""
    repository.add(
        FraudDecisionWrite(
            transaction_id=transaction_id,
            event_time=datetime(
                2018,
                6,
                15,
                10,
                transaction_id,
            ),
            customer_id=100,
            terminal_id=500,
            amount=amount,
            raw_probability=probability,
            calibrated_probability=probability,
            decision=decision,
            model_version=(
                "day-03-hgb-balanced-v1"
            ),
            policy_version=(
                "day-04-three-way-policy-v1"
            ),
        )
    )


def test_empty_repository_returns_zero_metrics(
    repository,
):
    metrics = build_operational_metrics(
        repository
    )

    assert metrics.total_transactions == 0
    assert metrics.approve_count == 0
    assert metrics.review_count == 0
    assert metrics.block_count == 0

    assert metrics.approve_rate == 0.0
    assert metrics.review_rate == 0.0
    assert metrics.block_rate == 0.0

    assert (
        metrics.average_calibrated_probability
        == 0.0
    )

    assert metrics.high_risk_rate == 0.0
    assert metrics.total_transaction_amount == 0.0
    assert metrics.average_transaction_amount == 0.0


def test_operational_metrics_are_correct(
    repository,
):
    add_decision(
        repository,
        transaction_id=1,
        amount=25.0,
        probability=0.02,
        decision="APPROVE",
    )

    add_decision(
        repository,
        transaction_id=2,
        amount=125.0,
        probability=0.06,
        decision="REVIEW",
    )

    add_decision(
        repository,
        transaction_id=3,
        amount=250.0,
        probability=0.12,
        decision="BLOCK",
    )

    metrics = build_operational_metrics(
        repository
    )

    assert metrics.total_transactions == 3

    assert metrics.approve_count == 1
    assert metrics.review_count == 1
    assert metrics.block_count == 1

    assert metrics.approve_rate == pytest.approx(
        1 / 3
    )
    assert metrics.review_rate == pytest.approx(
        1 / 3
    )
    assert metrics.block_rate == pytest.approx(
        1 / 3
    )

    assert (
        metrics.average_calibrated_probability
        == pytest.approx(
            (0.02 + 0.06 + 0.12) / 3
        )
    )

    assert metrics.high_risk_rate == pytest.approx(
        2 / 3
    )

    assert (
        metrics.total_transaction_amount
        == pytest.approx(400.0)
    )

    assert (
        metrics.average_transaction_amount
        == pytest.approx(400.0 / 3)
    )


def test_metrics_do_not_modify_repository(
    repository,
):
    add_decision(
        repository,
        transaction_id=1,
        amount=50.0,
        probability=0.02,
        decision="APPROVE",
    )

    before = repository.count()

    build_operational_metrics(repository)

    after = repository.count()

    assert before == after == 1