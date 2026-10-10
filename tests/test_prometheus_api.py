"""Tests for FraudShield Prometheus HTTP exposition."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fraudshield.api.app import create_app
from fraudshield.monitoring import (
    DataQualityMonitor,
    ModelDecisionMonitor,
    PrometheusExporter,
    RuntimeMonitor,
)


def build_exporter() -> PrometheusExporter:
    """Build deterministic monitoring state."""
    runtime = RuntimeMonitor()
    quality = DataQualityMonitor()
    decisions = ModelDecisionMonitor()

    runtime.record_success(
        decision="APPROVE",
        processing_time_seconds=0.2,
    )

    decisions.record(
        calibrated_probability=0.02,
        decision="APPROVE",
    )

    return PrometheusExporter(
        runtime_monitor=runtime,
        data_quality_monitor=quality,
        decision_monitor=decisions,
    )


def test_metrics_endpoint_returns_prometheus_payload() -> None:
    exporter = build_exporter()

    app = create_app(
        prometheus_dependency=lambda: exporter,
    )

    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200

    assert (
        "text/plain"
        in response.headers["content-type"]
    )

    assert (
        "fraudshield_requests_total 1.0"
        in response.text
    )

    assert (
        "fraudshield_predictions_total 1.0"
        in response.text
    )

    assert (
        "fraudshield_approve_total 1.0"
        in response.text
    )


def test_metrics_endpoint_fails_safely_when_unconfigured() -> None:
    app = create_app()

    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Prometheus monitoring is unavailable."
        )
    }


def test_metrics_endpoint_preserves_observability_headers() -> None:
    exporter = build_exporter()

    app = create_app(
        prometheus_dependency=lambda: exporter,
    )

    client = TestClient(app)

    response = client.get(
        "/metrics",
        headers={
            "X-Request-ID": "cp05-prometheus-test",
        },
    )

    assert response.status_code == 200

    assert (
        response.headers["X-Request-ID"]
        == "cp05-prometheus-test"
    )

    assert "X-Process-Time-Ms" in response.headers