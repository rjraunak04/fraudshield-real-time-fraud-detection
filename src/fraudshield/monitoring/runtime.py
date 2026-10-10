"""Runtime operational monitoring for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter


@dataclass(frozen=True, slots=True)
class RuntimeMetricsSnapshot:
    """Immutable snapshot of real-time platform activity."""

    requests_total: int
    requests_succeeded: int
    requests_failed: int
    approve_total: int
    review_total: int
    block_total: int
    duplicate_total: int
    processing_time_seconds_total: float

    @property
    def success_rate(self) -> float:
        """Return successful request fraction."""
        if self.requests_total == 0:
            return 0.0
        return self.requests_succeeded / self.requests_total

    @property
    def failure_rate(self) -> float:
        """Return failed request fraction."""
        if self.requests_total == 0:
            return 0.0
        return self.requests_failed / self.requests_total

    @property
    def average_processing_time_seconds(self) -> float:
        """Return average request processing duration."""
        if self.requests_total == 0:
            return 0.0
        return (
            self.processing_time_seconds_total
            / self.requests_total
        )


class RuntimeMonitor:
    """Thread-safe in-process runtime monitoring."""

    def __init__(self) -> None:
        self._lock = Lock()

        self._requests_total = 0
        self._requests_succeeded = 0
        self._requests_failed = 0

        self._approve_total = 0
        self._review_total = 0
        self._block_total = 0

        self._duplicate_total = 0
        self._processing_time_seconds_total = 0.0

    def record_success(
        self,
        *,
        decision: str,
        processing_time_seconds: float,
    ) -> None:
        """Record one successfully processed transaction."""
        if processing_time_seconds < 0:
            raise ValueError(
                "processing_time_seconds must be non-negative."
            )

        normalized = decision.upper()

        if normalized not in {
            "APPROVE",
            "REVIEW",
            "BLOCK",
        }:
            raise ValueError(
                f"Unsupported decision: {decision}."
            )

        with self._lock:
            self._requests_total += 1
            self._requests_succeeded += 1

            if normalized == "APPROVE":
                self._approve_total += 1
            elif normalized == "REVIEW":
                self._review_total += 1
            else:
                self._block_total += 1

            self._processing_time_seconds_total += (
                processing_time_seconds
            )

    def record_failure(
        self,
        *,
        processing_time_seconds: float,
    ) -> None:
        """Record one failed transaction."""
        if processing_time_seconds < 0:
            raise ValueError(
                "processing_time_seconds must be non-negative."
            )

        with self._lock:
            self._requests_total += 1
            self._requests_failed += 1
            self._processing_time_seconds_total += (
                processing_time_seconds
            )

    def record_duplicate(self) -> None:
        """Record one duplicate transaction attempt."""
        with self._lock:
            self._duplicate_total += 1

    def snapshot(self) -> RuntimeMetricsSnapshot:
        """Return an immutable consistent metrics snapshot."""
        with self._lock:
            return RuntimeMetricsSnapshot(
                requests_total=self._requests_total,
                requests_succeeded=self._requests_succeeded,
                requests_failed=self._requests_failed,
                approve_total=self._approve_total,
                review_total=self._review_total,
                block_total=self._block_total,
                duplicate_total=self._duplicate_total,
                processing_time_seconds_total=(
                    self._processing_time_seconds_total
                ),
            )


class ProcessingTimer:
    """Small monotonic timer for processing measurements."""

    def __init__(self) -> None:
        self._started_at: float | None = None

    def start(self) -> None:
        """Start the timer."""
        self._started_at = perf_counter()

    def elapsed_seconds(self) -> float:
        """Return elapsed monotonic duration."""
        if self._started_at is None:
            raise RuntimeError(
                "ProcessingTimer has not been started."
            )

        return perf_counter() - self._started_at