"""Integration tests for live API monitoring."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from fraudshield.api.app import create_app
from fraudshield.decision import RiskDecisionEngine
from fraudshield.monitoring import (
    DataQualityMonitor,
    FraudMonitoringService,
    ModelDecisionMonitor,
    PrometheusExporter,
    RuntimeMonitor,
)
from fraudshield.realtime import RealTimeFraudResult
from fraudshield.scoring.service import FraudScore
from fraudshield.streaming.events import TransactionEvent


@dataclass
class FakeRealTimeService:
    """Deterministic real-time service for monitoring tests."""

    probability: float = 0.06

    def process(
        self,
        event: TransactionEvent,
    ) -> RealTimeFraudResult:
        threshold = 0.048

        score = FraudScore(
            raw_probability=self.probability,
            calibrated_probability=self.probability,
            threshold=threshold,
            is_high_risk=(
                self.probability >= threshold
            ),
            model_version="day-03-hgb-balanced-v1",
            calibration_method="sigmoid",
        )

        decision = RiskDecisionEngine().decide(score)

        return RealTimeFraudResult(
            transaction_id=event.transaction_id,
            score=score,
            decision=decision,
        )


def test_raw_transaction_updates_prometheus_metrics() -> None:
    runtime = RuntimeMonitor()
    quality = DataQualityMonitor()
    decisions = ModelDecisionMonitor()

    monitoring_service = FraudMonitoringService(
        runtime_monitor=runtime,
        data_quality_monitor=quality,
        decision_monitor=decisions,
    )

    exporter = PrometheusExporter(
        runtime_monitor=runtime,
        data_quality_monitor=quality,
        decision_monitor=decisions,
    )

    realtime_service = FakeRealTimeService()

    app = create_app(
        realtime_dependency=lambda: realtime_service,
        monitoring_dependency=lambda: monitoring_service,
        prometheus_dependency=lambda: exporter,
    )

    client = TestClient(app)

    response = client.post(
        "/v1/transactions/score",
        json={
            "transaction_id": 1001,
            "event_time": "2018-06-15T10:00:00",
            "customer_id": 101,
            "terminal_id": 501,
            "amount": 50.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["action"] == "REVIEW"

    metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200

    payload = metrics_response.text

    assert "fraudshield_requests_total 1.0" in payload
    assert (
        "fraudshield_requests_succeeded_total 1.0"
        in payload
    )
    assert (
        "fraudshield_requests_failed_total 0.0"
        in payload
    )
    assert (
        "fraudshield_data_quality_events_total 1.0"
        in payload
    )
    assert (
        "fraudshield_data_quality_invalid_total 0.0"
        in payload
    )
    assert "fraudshield_predictions_total 1.0" in payload
    assert "fraudshield_review_total 1.0" in payload