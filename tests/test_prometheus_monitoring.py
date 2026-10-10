"""Tests for Prometheus monitoring exposition."""

from __future__ import annotations

from datetime import datetime

from fraudshield.monitoring import (
    DataQualityMonitor,
    ModelDecisionMonitor,
    PrometheusExporter,
    RuntimeMonitor,
)
from fraudshield.streaming.events import TransactionEvent


def build_exporter() -> tuple[
    PrometheusExporter,
    RuntimeMonitor,
    DataQualityMonitor,
    ModelDecisionMonitor,
]:
    """Build isolated monitoring components."""
    runtime = RuntimeMonitor()
    quality = DataQualityMonitor()
    decisions = ModelDecisionMonitor()

    exporter = PrometheusExporter(
        runtime_monitor=runtime,
        data_quality_monitor=quality,
        decision_monitor=decisions,
    )

    return exporter, runtime, quality, decisions


def test_empty_metrics_render() -> None:
    exporter, _, _, _ = build_exporter()

    payload = exporter.render().decode()

    assert "fraudshield_requests_total 0.0" in payload
    assert "fraudshield_predictions_total 0.0" in payload
    assert "fraudshield_data_quality_invalid_rate 0.0" in payload


def test_runtime_metrics_are_exported() -> None:
    exporter, runtime, _, _ = build_exporter()

    runtime.record_success(
        decision="APPROVE",
        processing_time_seconds=0.2,
    )
    runtime.record_failure(
        processing_time_seconds=0.4,
    )
    runtime.record_duplicate()

    payload = exporter.render().decode()

    assert "fraudshield_requests_total 2.0" in payload
    assert "fraudshield_requests_succeeded_total 1.0" in payload
    assert "fraudshield_requests_failed_total 1.0" in payload
    assert "fraudshield_duplicate_attempts_total 1.0" in payload


def test_data_quality_metrics_are_exported() -> None:
    exporter, _, quality, _ = build_exporter()

    event = TransactionEvent(
        transaction_id=1,
        event_time=datetime(2018, 6, 15, 10, 0),
        customer_id=101,
        terminal_id=501,
        amount=50.0,
    )

    quality.inspect(
        event,
        observed_at=datetime(2018, 6, 15, 10, 0),
    )
    quality.inspect(
        event,
        observed_at=datetime(2018, 6, 15, 10, 0),
    )

    payload = exporter.render().decode()

    assert "fraudshield_data_quality_events_total 2.0" in payload
    assert "fraudshield_data_quality_invalid_total 1.0" in payload
    assert "fraudshield_duplicate_transactions_total 1.0" in payload


def test_decision_metrics_are_exported() -> None:
    exporter, _, _, decisions = build_exporter()

    decisions.record(
        calibrated_probability=0.02,
        decision="APPROVE",
    )
    decisions.record(
        calibrated_probability=0.06,
        decision="REVIEW",
    )
    decisions.record(
        calibrated_probability=0.12,
        decision="BLOCK",
    )

    payload = exporter.render().decode()

    assert "fraudshield_predictions_total 3.0" in payload
    assert "fraudshield_approve_total 1.0" in payload
    assert "fraudshield_review_total 1.0" in payload
    assert "fraudshield_block_total 1.0" in payload