"""Failure-ordering tests for production transaction processing."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from fraudshield.features.online_pipeline import OnlineFeaturePipeline
from fraudshield.realtime import RealTimeFraudPipeline
from fraudshield.scoring.service import FraudScore
from fraudshield.state.store import InMemoryStateStore
from fraudshield.streaming.events import TransactionEvent


class FailingScorer:
    def score_one(self, features):
        raise RuntimeError("model unavailable")


class FixedScorer:
    def score_one(self, features):
        return FraudScore(
            raw_probability=0.01,
            calibrated_probability=0.01,
            model_version="test-model",
        )


class FailingStateStore(InMemoryStateStore):
    def record(self, event):
        raise RuntimeError("state unavailable")


def _event():
    return TransactionEvent(
        transaction_id=800001,
        event_time=datetime(2018, 6, 15, 12, 0),
        customer_id=10,
        terminal_id=20,
        amount=25.0,
    )


def test_scoring_failure_does_not_write_database_or_state():
    state = InMemoryStateStore()
    repository = MagicMock()
    session = MagicMock()
    pipeline = RealTimeFraudPipeline(
        feature_pipeline=OnlineFeaturePipeline(state),
        scorer=FailingScorer(),
        decision_engine=MagicMock(),
        repository=repository,
        session=session,
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        pipeline.process(_event())

    repository.add.assert_not_called()
    session.commit.assert_not_called()
    assert state.customer_history(
        10,
        datetime(2018, 6, 15, 12, 1),
    ).count == 0


def test_database_failure_rolls_back_and_does_not_advance_state():
    state = InMemoryStateStore()
    repository = MagicMock()
    repository.add.side_effect = RuntimeError("database unavailable")
    session = MagicMock()
    pipeline = RealTimeFraudPipeline(
        feature_pipeline=OnlineFeaturePipeline(state),
        scorer=FixedScorer(),
        decision_engine=MagicMock(
            decide=MagicMock(
                side_effect=RuntimeError("decision stub must be replaced")
            )
        ),
        repository=repository,
        session=session,
    )

    from fraudshield.decision import RiskDecisionEngine

    pipeline._decision_engine = RiskDecisionEngine()

    with pytest.raises(RuntimeError, match="database unavailable"):
        pipeline.process(_event())

    session.rollback.assert_called_once()
    session.commit.assert_not_called()
    assert state.customer_history(
        10,
        datetime(2018, 6, 15, 12, 1),
    ).count == 0


def test_state_failure_occurs_after_durable_database_commit():
    state = FailingStateStore()
    repository = MagicMock()
    session = MagicMock()

    from fraudshield.decision import RiskDecisionEngine

    pipeline = RealTimeFraudPipeline(
        feature_pipeline=OnlineFeaturePipeline(state),
        scorer=FixedScorer(),
        decision_engine=RiskDecisionEngine(),
        repository=repository,
        session=session,
    )

    with pytest.raises(RuntimeError, match="state unavailable"):
        pipeline.process(_event())

    repository.add.assert_called_once()
    session.commit.assert_called_once()
    session.rollback.assert_not_called()
