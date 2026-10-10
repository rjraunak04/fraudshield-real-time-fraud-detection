"""Tests for streaming data-quality monitoring."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fraudshield.monitoring import DataQualityMonitor
from fraudshield.streaming.events import TransactionEvent

BASE_TIME = datetime(2018, 6, 15, 10, 0)


def make_event(
    transaction_id: int,
    *,
    minute: int = 0,
    amount: float = 50.0,
) -> TransactionEvent:
    """Create a deterministic valid transaction."""
    return TransactionEvent(
        transaction_id=transaction_id,
        event_time=BASE_TIME
        + timedelta(minutes=minute),
        customer_id=101,
        terminal_id=501,
        amount=amount,
    )


def test_empty_snapshot() -> None:
    monitor = DataQualityMonitor()

    snapshot = monitor.snapshot()

    assert snapshot.events_total == 0
    assert snapshot.valid_events == 0
    assert snapshot.invalid_events == 0
    assert snapshot.valid_rate == 0.0
    assert snapshot.invalid_rate == 0.0


def test_valid_event_is_recorded() -> None:
    monitor = DataQualityMonitor()

    result = monitor.inspect(
        make_event(1),
        observed_at=BASE_TIME,
    )

    assert result.is_valid is True
    assert result.violations == ()

    snapshot = monitor.snapshot()

    assert snapshot.events_total == 1
    assert snapshot.valid_events == 1
    assert snapshot.invalid_events == 0
    assert snapshot.valid_rate == 1.0


def test_duplicate_transaction_is_detected() -> None:
    monitor = DataQualityMonitor()

    event = make_event(1)

    first = monitor.inspect(
        event,
        observed_at=BASE_TIME,
    )
    second = monitor.inspect(
        event,
        observed_at=BASE_TIME,
    )

    assert first.is_valid is True
    assert second.is_valid is False
    assert (
        "DUPLICATE_TRANSACTION"
        in second.violations
    )

    snapshot = monitor.snapshot()

    assert snapshot.events_total == 2
    assert snapshot.valid_events == 1
    assert snapshot.invalid_events == 1
    assert (
        snapshot.duplicate_transaction_total
        == 1
    )


def test_future_timestamp_is_detected() -> None:
    monitor = DataQualityMonitor()

    event = make_event(
        1,
        minute=10,
    )

    result = monitor.inspect(
        event,
        observed_at=BASE_TIME,
    )

    assert result.is_valid is False
    assert "FUTURE_TIMESTAMP" in result.violations

    assert (
        monitor.snapshot().future_timestamp_total
        == 1
    )


def test_event_time_regression_is_detected() -> None:
    monitor = DataQualityMonitor()

    monitor.inspect(
        make_event(1, minute=10),
        observed_at=BASE_TIME
        + timedelta(hours=1),
    )

    result = monitor.inspect(
        make_event(2, minute=5),
        observed_at=BASE_TIME
        + timedelta(hours=1),
    )

    assert result.is_valid is False
    assert (
        "EVENT_TIME_REGRESSION"
        in result.violations
    )

    assert (
        monitor.snapshot()
        .event_time_regression_total
        == 1
    )


def test_multiple_violations_count_event_once() -> None:
    monitor = DataQualityMonitor()

    monitor.inspect(
        make_event(1, minute=10),
        observed_at=BASE_TIME,
    )

    result = monitor.inspect(
        make_event(1, minute=5),
        observed_at=BASE_TIME,
    )

    assert result.is_valid is False
    assert set(result.violations) == {
        "DUPLICATE_TRANSACTION",
        "FUTURE_TIMESTAMP",
        "EVENT_TIME_REGRESSION",
    }

    snapshot = monitor.snapshot()

    assert snapshot.events_total == 2
    assert snapshot.invalid_events == 2

    assert (
        snapshot.duplicate_transaction_total
        == 1
    )
    assert snapshot.future_timestamp_total == 2
    assert (
        snapshot.event_time_regression_total
        == 1
    )


def test_valid_and_invalid_rates() -> None:
    monitor = DataQualityMonitor()

    monitor.inspect(
        make_event(1),
        observed_at=BASE_TIME,
    )

    monitor.inspect(
        make_event(2, minute=1),
        observed_at=BASE_TIME,
    )

    snapshot = monitor.snapshot()

    assert snapshot.events_total == 2
    assert snapshot.valid_events == 1
    assert snapshot.invalid_events == 1

    assert snapshot.valid_rate == pytest.approx(
        0.5
    )
    assert snapshot.invalid_rate == pytest.approx(
        0.5
    )