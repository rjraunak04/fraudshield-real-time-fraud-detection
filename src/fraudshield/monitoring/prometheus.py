"""Prometheus exposition for FraudShield monitoring."""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    generate_latest,
)

from fraudshield.monitoring.decisions import ModelDecisionMonitor
from fraudshield.monitoring.quality import DataQualityMonitor
from fraudshield.monitoring.runtime import RuntimeMonitor

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class PrometheusExporter:
    """Build Prometheus metrics from monitoring snapshots."""

    def __init__(
        self,
        *,
        runtime_monitor: RuntimeMonitor,
        data_quality_monitor: DataQualityMonitor,
        decision_monitor: ModelDecisionMonitor,
    ) -> None:
        self._runtime_monitor = runtime_monitor
        self._data_quality_monitor = data_quality_monitor
        self._decision_monitor = decision_monitor

    def render(self) -> bytes:
        """Render the current monitoring state."""
        registry = CollectorRegistry()

        runtime = self._runtime_monitor.snapshot()
        quality = self._data_quality_monitor.snapshot()
        decisions = self._decision_monitor.snapshot()

        self._gauge(
            registry,
            "fraudshield_requests_total",
            "Transactions observed by runtime monitoring.",
            runtime.requests_total,
        )
        self._gauge(
            registry,
            "fraudshield_requests_succeeded_total",
            "Successfully processed transactions.",
            runtime.requests_succeeded,
        )
        self._gauge(
            registry,
            "fraudshield_requests_failed_total",
            "Failed transaction processing attempts.",
            runtime.requests_failed,
        )
        self._gauge(
            registry,
            "fraudshield_duplicate_attempts_total",
            "Duplicate transaction attempts.",
            runtime.duplicate_total,
        )
        self._gauge(
            registry,
            "fraudshield_average_processing_seconds",
            "Average transaction processing duration.",
            runtime.average_processing_time_seconds,
        )

        self._gauge(
            registry,
            "fraudshield_data_quality_events_total",
            "Events inspected by data-quality monitoring.",
            quality.events_total,
        )
        self._gauge(
            registry,
            "fraudshield_data_quality_invalid_total",
            "Invalid events observed by data-quality monitoring.",
            quality.invalid_events,
        )
        self._gauge(
            registry,
            "fraudshield_data_quality_invalid_rate",
            "Fraction of inspected events with quality violations.",
            quality.invalid_rate,
        )
        self._gauge(
            registry,
            "fraudshield_duplicate_transactions_total",
            "Duplicate transactions detected by data-quality monitoring.",
            quality.duplicate_transaction_total,
        )
        self._gauge(
            registry,
            "fraudshield_future_timestamps_total",
            "Future timestamps detected by data-quality monitoring.",
            quality.future_timestamp_total,
        )
        self._gauge(
            registry,
            "fraudshield_event_time_regressions_total",
            "Event-time regressions detected by data-quality monitoring.",
            quality.event_time_regression_total,
        )

        self._gauge(
            registry,
            "fraudshield_predictions_total",
            "Predictions observed by model monitoring.",
            decisions.predictions_total,
        )
        self._gauge(
            registry,
            "fraudshield_approve_total",
            "APPROVE decisions observed.",
            decisions.approve_total,
        )
        self._gauge(
            registry,
            "fraudshield_review_total",
            "REVIEW decisions observed.",
            decisions.review_total,
        )
        self._gauge(
            registry,
            "fraudshield_block_total",
            "BLOCK decisions observed.",
            decisions.block_total,
        )
        self._gauge(
            registry,
            "fraudshield_high_risk_rate",
            "Fraction of REVIEW plus BLOCK decisions.",
            decisions.high_risk_rate,
        )
        self._gauge(
            registry,
            "fraudshield_mean_calibrated_probability",
            "Mean calibrated fraud probability.",
            decisions.mean_probability,
        )
        self._gauge(
            registry,
            "fraudshield_near_review_threshold_total",
            "Predictions near the review threshold.",
            decisions.near_review_threshold_total,
        )
        self._gauge(
            registry,
            "fraudshield_near_block_threshold_total",
            "Predictions near the block threshold.",
            decisions.near_block_threshold_total,
        )

        return generate_latest(registry)

    @staticmethod
    def _gauge(
        registry: CollectorRegistry,
        name: str,
        description: str,
        value: float | int,
    ) -> None:
        """Register and set one snapshot gauge."""
        metric = Gauge(
            name,
            description,
            registry=registry,
        )
        metric.set(value)