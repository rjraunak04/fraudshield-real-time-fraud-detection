"""Tests for application-level fraud monitoring orchestration."""

from __future__ import annotations

from datetime import datetime

from fraudshield.decision import RiskDecisionEngine
from fraudshield.monitoring import (
    DataQualityMonitor,
    FraudMonitoringService,
    ModelDecisionMonitor,
    RuntimeMonitor,
)
from fraudshield.realtime import RealTimeFraudResult
from fraudshield.scoring.service import FraudScore
from fraudshield.streaming.events import TransactionEvent


def build_components() -> tuple[
    FraudMonitoringService,
    RuntimeMonitor,
    DataQualityMonitor,
    ModelDecisionMonitor,
]:
    """Build isolated monitoring components."""
    runtime = RuntimeMonitor()
    quality = DataQualityMonitor()
    decisions = ModelDecisionMonitor()

    service = FraudMonitoringService(
        runtime_monitor=runtime,
        data_quality_monitor=quality,
        decision_monitor=decisions,
    )

    return service, runtime, quality, decisions


def build_event() -> TransactionEvent:
    """Build one valid raw transaction."""
    return TransactionEvent(
        transaction_id=1,
        event_time=datetime(2018, 6, 15, 10, 0),
        customer_id=101,
        terminal_id=501,
        amount=50.0,
    )


def build_result(
    probability: float = 0.06,
) -> RealTimeFraudResult:
    """Build one deterministic real-time result."""
    threshold = 0.048

    score = FraudScore(
        raw_probability=probability,
        calibrated_probability=probability,
        threshold=threshold,
        is_high_risk=probability >= threshold,
        model_version="day-03-hgb-balanced-v1",
        calibration_method="sigmoid",
    )

    decision = RiskDecisionEngine().decide(score)

    return RealTimeFraudResult(
        transaction_id=1,
        score=score,
        decision=decision,
    )


def test_inspect_event_updates_data_quality() -> None:
    service, _, quality, _ = build_components()

    service.inspect_event(
        build_event(),
        observed_at=datetime(2018, 6, 15, 10, 0),
    )

    snapshot = quality.snapshot()

    assert snapshot.events_total == 1
    assert snapshot.valid_events == 1
    assert snapshot.invalid_events == 0


def test_success_updates_runtime_and_decision_monitoring() -> None:
    service, runtime, _, decisions = build_components()

    service.record_success(
        build_result(),
        processing_time_seconds=0.25,
    )

    runtime_snapshot = runtime.snapshot()
    decision_snapshot = decisions.snapshot()

    assert runtime_snapshot.requests_total == 1
    assert runtime_snapshot.requests_succeeded == 1
    assert runtime_snapshot.review_total == 1

    assert decision_snapshot.predictions_total == 1
    assert decision_snapshot.review_total == 1


def test_failure_updates_runtime_monitoring() -> None:
    service, runtime, _, decisions = build_components()

    service.record_failure(
        processing_time_seconds=0.4,
    )

    runtime_snapshot = runtime.snapshot()

    assert runtime_snapshot.requests_total == 1
    assert runtime_snapshot.requests_failed == 1
    assert decisions.snapshot().predictions_total == 0


def test_duplicate_updates_duplicate_counter() -> None:
    service, runtime, _, _ = build_components()

    service.record_duplicate()

    assert runtime.snapshot().duplicate_total == 1