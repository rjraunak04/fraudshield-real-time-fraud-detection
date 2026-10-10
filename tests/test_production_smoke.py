"""End-to-end smoke test across FraudShield runtime components."""

from datetime import datetime

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from fraudshield.decision import DecisionAction, RiskDecisionEngine
from fraudshield.features.online_pipeline import OnlineFeaturePipeline
from fraudshield.monitoring import (
    DataQualityMonitor,
    FraudMonitoringService,
    ModelDecisionMonitor,
    PrometheusExporter,
    RuntimeMonitor,
)
from fraudshield.persistence import (
    FraudDecisionRepository,
    create_schema,
    create_session_factory,
)
from fraudshield.realtime import RealTimeFraudPipeline
from fraudshield.scoring.service import FraudScorer
from fraudshield.state.store import InMemoryStateStore
from fraudshield.streaming.events import TransactionEvent


class FixedModel:
    """Deterministic model used only for infrastructure verification."""

    def predict_proba(self, frame):
        probability = np.full(len(frame), 0.06, dtype=float)
        return np.column_stack((1.0 - probability, probability))


class IdentityCalibrator:
    """Return the supplied probability unchanged."""

    def predict(self, probabilities):
        return np.asarray(probabilities, dtype=float)


def test_transaction_reaches_decision_storage_state_and_monitoring():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    session = create_session_factory(engine)()
    state = InMemoryStateStore()
    repository = FraudDecisionRepository(session)

    runtime = RuntimeMonitor()
    quality = DataQualityMonitor()
    decisions = ModelDecisionMonitor()
    monitoring = FraudMonitoringService(
        runtime_monitor=runtime,
        data_quality_monitor=quality,
        decision_monitor=decisions,
    )
    exporter = PrometheusExporter(
        runtime_monitor=runtime,
        data_quality_monitor=quality,
        decision_monitor=decisions,
    )

    pipeline = RealTimeFraudPipeline(
        feature_pipeline=OnlineFeaturePipeline(state),
        scorer=FraudScorer(
            model=FixedModel(),
            calibrator=IdentityCalibrator(),
        ),
        decision_engine=RiskDecisionEngine(),
        repository=repository,
        session=session,
    )
    event = TransactionEvent(
        transaction_id=700001,
        event_time=datetime(2018, 6, 15, 14, 30),
        customer_id=101,
        terminal_id=501,
        amount=125.50,
    )

    try:
        monitoring.inspect_event(
            event,
            observed_at=datetime(2018, 6, 15, 14, 30),
        )
        result = pipeline.process(event)
        monitoring.record_success(
            result,
            processing_time_seconds=0.01,
        )

        assert result.decision.action is DecisionAction.REVIEW
        assert result.score.calibrated_probability == 0.06
        assert result.decision.review_threshold == 0.048
        assert result.decision.block_threshold == 0.100

        stored = repository.get_by_transaction_id(event.transaction_id)
        assert stored is not None
        assert stored.decision == "REVIEW"
        assert stored.model_version == "day-03-hgb-balanced-v1"
        assert stored.policy_version == "day-04-three-way-policy-v1"

        history = state.customer_history(
            event.customer_id,
            datetime(2018, 6, 15, 14, 31),
        )
        assert history.count == 1

        metrics = exporter.render().decode("utf-8")
        assert "fraudshield_requests_succeeded_total 1.0" in metrics
        assert "fraudshield_predictions_total 1.0" in metrics
        assert "fraudshield_review_total 1.0" in metrics
    finally:
        session.close()
        engine.dispose()
