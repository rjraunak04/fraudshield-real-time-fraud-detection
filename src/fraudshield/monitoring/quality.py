"""Streaming data-quality monitoring for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from threading import Lock

from fraudshield.streaming.events import TransactionEvent


@dataclass(frozen=True, slots=True)
class DataQualityResult:
    """Quality assessment for one transaction event."""

    is_valid: bool
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DataQualitySnapshot:
    """Immutable aggregate data-quality snapshot."""

    events_total: int
    valid_events: int
    invalid_events: int
    invalid_identifier_total: int
    invalid_amount_total: int
    non_finite_amount_total: int
    duplicate_transaction_total: int
    future_timestamp_total: int
    event_time_regression_total: int

    @property
    def valid_rate(self) -> float:
        """Return fraction of valid events."""
        if self.events_total == 0:
            return 0.0
        return self.valid_events / self.events_total

    @property
    def invalid_rate(self) -> float:
        """Return fraction of invalid events."""
        if self.events_total == 0:
            return 0.0
        return self.invalid_events / self.events_total


class DataQualityMonitor:
    """Thread-safe streaming transaction quality monitor."""

    def __init__(self) -> None:
        self._lock = Lock()

        self._events_total = 0
        self._valid_events = 0
        self._invalid_events = 0

        self._invalid_identifier_total = 0
        self._invalid_amount_total = 0
        self._non_finite_amount_total = 0
        self._duplicate_transaction_total = 0
        self._future_timestamp_total = 0
        self._event_time_regression_total = 0

        self._seen_transaction_ids: set[int] = set()
        self._latest_event_time: datetime | None = None

    def inspect(
        self,
        event: TransactionEvent,
        *,
        observed_at: datetime | None = None,
    ) -> DataQualityResult:
        """Inspect and record data quality for one event."""
        if observed_at is None:
            observed_at = datetime.now(
                tz=event.event_time.tzinfo
            )

        violations: list[str] = []

        with self._lock:
            if (
                event.transaction_id < 0
                or event.customer_id < 0
                or event.terminal_id < 0
            ):
                violations.append(
                    "INVALID_IDENTIFIER"
                )
                self._invalid_identifier_total += 1

            amount = float(event.amount)

            if not isfinite(amount):
                violations.append(
                    "NON_FINITE_AMOUNT"
                )
                self._non_finite_amount_total += 1
            elif amount < 0:
                violations.append(
                    "INVALID_AMOUNT"
                )
                self._invalid_amount_total += 1

            if (
                event.transaction_id
                in self._seen_transaction_ids
            ):
                violations.append(
                    "DUPLICATE_TRANSACTION"
                )
                self._duplicate_transaction_total += 1

            if event.event_time > observed_at:
                violations.append(
                    "FUTURE_TIMESTAMP"
                )
                self._future_timestamp_total += 1

            if (
                self._latest_event_time is not None
                and event.event_time
                < self._latest_event_time
            ):
                violations.append(
                    "EVENT_TIME_REGRESSION"
                )
                self._event_time_regression_total += 1

            self._events_total += 1

            if violations:
                self._invalid_events += 1
            else:
                self._valid_events += 1

            self._seen_transaction_ids.add(
                event.transaction_id
            )

            if (
                self._latest_event_time is None
                or event.event_time
                > self._latest_event_time
            ):
                self._latest_event_time = (
                    event.event_time
                )

        return DataQualityResult(
            is_valid=not violations,
            violations=tuple(violations),
        )

    def snapshot(self) -> DataQualitySnapshot:
        """Return an immutable quality snapshot."""
        with self._lock:
            return DataQualitySnapshot(
                events_total=self._events_total,
                valid_events=self._valid_events,
                invalid_events=self._invalid_events,
                invalid_identifier_total=(
                    self._invalid_identifier_total
                ),
                invalid_amount_total=(
                    self._invalid_amount_total
                ),
                non_finite_amount_total=(
                    self._non_finite_amount_total
                ),
                duplicate_transaction_total=(
                    self._duplicate_transaction_total
                ),
                future_timestamp_total=(
                    self._future_timestamp_total
                ),
                event_time_regression_total=(
                    self._event_time_regression_total
                ),
            )