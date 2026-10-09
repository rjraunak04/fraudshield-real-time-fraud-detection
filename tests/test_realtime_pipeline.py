"""Tests for the end-to-end FraudShield real-time pipeline."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from sqlalchemy.orm import Session

from fraudshield.decision import DecisionAction, RiskDecisionEngine
from fraudshield.features.online_pipeline import OnlineFeaturePipeline
from fraudshield.persistence import (
    DuplicateDecisionError,
    FraudDecisionRepository,
    create_database_engine,
    create_schema,
    create_session_factory,
)
from fraudshield.realtime import RealTimeFraudPipeline
from fraudshield.scoring.service import FraudScorer
from fraudshield.state.store import InMemoryStateStore
from fraudshield.streaming.events import TransactionEvent


class FixedModel:
    """Deterministic probability model for integration tests."""

    def __init__(self, probability: float) -> None:
        self._probability = probability

    def predict_proba(self, X):
        probability = self._probability
        return np.array(
            [[1.0 - probability, probability]],
            dtype=float,
        )


class IdentityCalibrator:
    """Return raw probability unchanged."""

    def predict(self, probabilities):
        return np.asarray(probabilities, dtype=float)


class FailingModel:
    """Simulate model-serving failure."""

    def predict_proba(self, X):
        raise RuntimeError("scoring failed")


def make_event(
    transaction_id: int = 1,
) -> TransactionEvent:
    """Build one deterministic raw transaction."""
    return TransactionEvent(
        transaction_id=transaction_id,
        event_time=datetime(2018, 4, 1, 12, 0, 0),
        customer_id=101,
        terminal_id=501,
        amount=125.50,
    )


def build_pipeline(
    session: Session,
    state: InMemoryStateStore,
    *,
    probability: float = 0.060,
    model=None,
) -> tuple[
    RealTimeFraudPipeline,
    FraudDecisionRepository,
]:
    """Build an isolated real-time pipeline."""
    scorer = FraudScorer(
        model=model or FixedModel(probability),
        calibrator=IdentityCalibrator(),
    )

    repository = FraudDecisionRepository(session)

    pipeline = RealTimeFraudPipeline(
        feature_pipeline=OnlineFeaturePipeline(state),
        scorer=scorer,
        decision_engine=RiskDecisionEngine(),
        repository=repository,
        session=session,
    )

    return pipeline, repository


@pytest.fixture
def database_session() -> Session:
    """Create isolated persistence storage."""
    engine = create_database_engine(
        "sqlite:///:memory:"
    )
    create_schema(engine)

    factory = create_session_factory(engine)
    session = factory()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_successful_transaction_is_persisted_and_committed_to_state(
    database_session: Session,
) -> None:
    state = InMemoryStateStore()
    pipeline, repository = build_pipeline(
        database_session,
        state,
    )
    event = make_event()

    result = pipeline.process(event)

    assert result.transaction_id == event.transaction_id
    assert result.decision.action is DecisionAction.REVIEW
    assert repository.count() == 1

    history = state.customer_history(
        event.customer_id,
        datetime(2018, 4, 1, 12, 1, 0),
    )

    assert history.count == 1


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.020, DecisionAction.APPROVE),
        (0.060, DecisionAction.REVIEW),
        (0.120, DecisionAction.BLOCK),
    ],
)
def test_pipeline_supports_three_way_decisions(
    database_session: Session,
    probability: float,
    expected: DecisionAction,
) -> None:
    state = InMemoryStateStore()
    pipeline, repository = build_pipeline(
        database_session,
        state,
        probability=probability,
    )

    result = pipeline.process(make_event())

    assert result.decision.action is expected

    stored = repository.get_by_transaction_id(1)

    assert stored is not None
    assert stored.decision == expected.value


def test_scoring_failure_does_not_persist_or_mutate_state(
    database_session: Session,
) -> None:
    state = InMemoryStateStore()
    pipeline, repository = build_pipeline(
        database_session,
        state,
        model=FailingModel(),
    )
    event = make_event()

    with pytest.raises(
        RuntimeError,
        match="scoring failed",
    ):
        pipeline.process(event)

    assert repository.count() == 0

    history = state.customer_history(
        event.customer_id,
        datetime(2018, 4, 1, 12, 1, 0),
    )

    assert history.count == 0


def test_duplicate_transaction_does_not_write_state_twice(
    database_session: Session,
) -> None:
    state = InMemoryStateStore()
    pipeline, repository = build_pipeline(
        database_session,
        state,
    )
    event = make_event()

    pipeline.process(event)

    with pytest.raises(DuplicateDecisionError):
        pipeline.process(event)

    assert repository.count() == 1

    history = state.customer_history(
        event.customer_id,
        datetime(2018, 4, 1, 12, 1, 0),
    )

    assert history.count == 1


def test_persisted_record_contains_frozen_versions(
    database_session: Session,
) -> None:
    state = InMemoryStateStore()
    pipeline, repository = build_pipeline(
        database_session,
        state,
    )

    pipeline.process(make_event())

    stored = repository.get_by_transaction_id(1)

    assert stored is not None
    assert stored.model_version == "day-03-hgb-balanced-v1"
    assert stored.policy_version == "day-04-three-way-policy-v1"


def test_current_event_is_not_visible_during_feature_generation(
    database_session: Session,
) -> None:
    state = InMemoryStateStore()
    pipeline, _ = build_pipeline(
        database_session,
        state,
    )

    first = make_event(1)

    pipeline.process(first)

    second = TransactionEvent(
        transaction_id=2,
        event_time=datetime(2018, 4, 1, 12, 5, 0),
        customer_id=101,
        terminal_id=501,
        amount=50.0,
    )

    pipeline.process(second)

    history = state.customer_history(
        101,
        datetime(2018, 4, 1, 12, 6, 0),
    )

    assert history.count == 2


def test_successful_result_preserves_frozen_threshold(
    database_session: Session,
) -> None:
    state = InMemoryStateStore()
    pipeline, _ = build_pipeline(
        database_session,
        state,
    )

    result = pipeline.process(make_event())

    assert result.score.threshold == pytest.approx(0.048)
    assert result.decision.review_threshold == pytest.approx(0.048)
    assert result.decision.block_threshold == pytest.approx(0.100)