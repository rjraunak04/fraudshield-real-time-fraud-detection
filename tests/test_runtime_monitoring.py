"""Tests for FraudShield runtime monitoring."""

from __future__ import annotations

import pytest

from fraudshield.monitoring import (
    ProcessingTimer,
    RuntimeMonitor,
)


def test_empty_runtime_snapshot() -> None:
    monitor = RuntimeMonitor()

    snapshot = monitor.snapshot()

    assert snapshot.requests_total == 0
    assert snapshot.requests_succeeded == 0
    assert snapshot.requests_failed == 0

    assert snapshot.approve_total == 0
    assert snapshot.review_total == 0
    assert snapshot.block_total == 0
    assert snapshot.duplicate_total == 0

    assert snapshot.success_rate == 0.0
    assert snapshot.failure_rate == 0.0
    assert (
        snapshot.average_processing_time_seconds
        == 0.0
    )


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("APPROVE", (1, 0, 0)),
        ("REVIEW", (0, 1, 0)),
        ("BLOCK", (0, 0, 1)),
    ],
)
def test_successful_decision_is_recorded(
    decision: str,
    expected: tuple[int, int, int],
) -> None:
    monitor = RuntimeMonitor()

    monitor.record_success(
        decision=decision,
        processing_time_seconds=0.25,
    )

    snapshot = monitor.snapshot()

    assert snapshot.requests_total == 1
    assert snapshot.requests_succeeded == 1
    assert snapshot.requests_failed == 0

    assert (
        snapshot.approve_total,
        snapshot.review_total,
        snapshot.block_total,
    ) == expected

    assert snapshot.success_rate == 1.0
    assert snapshot.failure_rate == 0.0
    assert (
        snapshot.average_processing_time_seconds
        == pytest.approx(0.25)
    )


def test_failure_is_recorded() -> None:
    monitor = RuntimeMonitor()

    monitor.record_failure(
        processing_time_seconds=0.5
    )

    snapshot = monitor.snapshot()

    assert snapshot.requests_total == 1
    assert snapshot.requests_failed == 1
    assert snapshot.failure_rate == 1.0


def test_duplicate_is_separate_metric() -> None:
    monitor = RuntimeMonitor()

    monitor.record_duplicate()
    monitor.record_duplicate()

    snapshot = monitor.snapshot()

    assert snapshot.duplicate_total == 2
    assert snapshot.requests_total == 0


def test_multiple_events_produce_correct_rates() -> None:
    monitor = RuntimeMonitor()

    monitor.record_success(
        decision="APPROVE",
        processing_time_seconds=0.1,
    )
    monitor.record_success(
        decision="REVIEW",
        processing_time_seconds=0.2,
    )
    monitor.record_failure(
        processing_time_seconds=0.3
    )

    snapshot = monitor.snapshot()

    assert snapshot.requests_total == 3
    assert snapshot.requests_succeeded == 2
    assert snapshot.requests_failed == 1

    assert snapshot.success_rate == pytest.approx(
        2 / 3
    )
    assert snapshot.failure_rate == pytest.approx(
        1 / 3
    )

    assert (
        snapshot.average_processing_time_seconds
        == pytest.approx(0.2)
    )


@pytest.mark.parametrize(
    "duration",
    [-0.1, -1.0],
)
def test_negative_success_duration_rejected(
    duration: float,
) -> None:
    monitor = RuntimeMonitor()

    with pytest.raises(ValueError):
        monitor.record_success(
            decision="APPROVE",
            processing_time_seconds=duration,
        )


def test_negative_failure_duration_rejected() -> None:
    monitor = RuntimeMonitor()

    with pytest.raises(ValueError):
        monitor.record_failure(
            processing_time_seconds=-0.1
        )


def test_unknown_decision_rejected() -> None:
    monitor = RuntimeMonitor()

    with pytest.raises(ValueError):
        monitor.record_success(
            decision="UNKNOWN",
            processing_time_seconds=0.1,
        )


def test_processing_timer_requires_start() -> None:
    timer = ProcessingTimer()

    with pytest.raises(RuntimeError):
        timer.elapsed_seconds()


def test_processing_timer_returns_non_negative_time() -> None:
    timer = ProcessingTimer()

    timer.start()

    assert timer.elapsed_seconds() >= 0.0