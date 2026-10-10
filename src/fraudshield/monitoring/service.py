"""Application-level monitoring orchestration."""

from __future__ import annotations

import logging
from datetime import datetime

from fraudshield.monitoring.decisions import ModelDecisionMonitor
from fraudshield.monitoring.quality import DataQualityMonitor
from fraudshield.monitoring.runtime import RuntimeMonitor
from fraudshield.realtime import RealTimeFraudResult
from fraudshield.streaming.events import TransactionEvent

logger = logging.getLogger("fraudshield.monitoring")


class FraudMonitoringService:
    """Coordinate monitoring without owning business processing."""

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

    def inspect_event(
        self,
        event: TransactionEvent,
        *,
        observed_at: datetime | None = None,
    ) -> None:
        """Inspect transaction data quality without breaking scoring."""
        try:
            self._data_quality_monitor.inspect(
                event,
                observed_at=observed_at,
            )
        except Exception:
            logger.exception(
                "Data-quality monitoring failed.",
            )

    def record_success(
        self,
        result: RealTimeFraudResult,
        *,
        processing_time_seconds: float,
    ) -> None:
        """Record successful processing and decision behaviour."""
        try:
            self._runtime_monitor.record_success(
                decision=result.decision.action.value,
                processing_time_seconds=processing_time_seconds,
            )

            self._decision_monitor.record(
                calibrated_probability=(
                    result.score.calibrated_probability
                ),
                decision=result.decision.action.value,
            )
        except Exception:
            logger.exception(
                "Success monitoring failed.",
            )

    def record_failure(
        self,
        *,
        processing_time_seconds: float,
    ) -> None:
        """Record failed transaction processing."""
        try:
            self._runtime_monitor.record_failure(
                processing_time_seconds=processing_time_seconds,
            )
        except Exception:
            logger.exception(
                "Failure monitoring failed.",
            )

    def record_duplicate(self) -> None:
        """Record duplicate transaction attempts."""
        try:
            self._runtime_monitor.record_duplicate()
        except Exception:
            logger.exception(
                "Duplicate monitoring failed.",
            )