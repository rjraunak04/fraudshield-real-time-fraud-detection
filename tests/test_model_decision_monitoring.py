"""Tests for model-output and decision monitoring."""

from __future__ import annotations

import math

import pytest

from fraudshield.monitoring import (
    ModelDecisionMonitor,
)


def test_empty_snapshot_is_zero_safe() -> None:
    monitor = ModelDecisionMonitor()

    snapshot = monitor.snapshot()

    assert snapshot.predictions_total == 0

    assert snapshot.approve_rate == 0.0
    assert snapshot.review_rate == 0.0
    assert snapshot.block_rate == 0.0
    assert snapshot.high_risk_rate == 0.0

    assert snapshot.mean_probability == 0.0

    assert snapshot.probability_min is None
    assert snapshot.probability_max is None


def test_three_decisions_are_aggregated() -> None:
    monitor = ModelDecisionMonitor()

    monitor.record(
        calibrated_probability=0.02,
        decision="APPROVE",
    )
    monitor.record(
        calibrated_probability=0.06,
        decision="REVIEW",
    )
    monitor.record(
        calibrated_probability=0.12,
        decision="BLOCK",
    )

    snapshot = monitor.snapshot()

    assert snapshot.predictions_total == 3

    assert snapshot.approve_total == 1
    assert snapshot.review_total == 1
    assert snapshot.block_total == 1

    assert snapshot.approve_rate == pytest.approx(
        1 / 3
    )
    assert snapshot.review_rate == pytest.approx(
        1 / 3
    )
    assert snapshot.block_rate == pytest.approx(
        1 / 3
    )

    assert snapshot.high_risk_rate == pytest.approx(
        2 / 3
    )

    assert snapshot.mean_probability == pytest.approx(
        0.20 / 3
    )

    assert snapshot.probability_min == pytest.approx(
        0.02
    )
    assert snapshot.probability_max == pytest.approx(
        0.12
    )


def test_review_threshold_proximity_is_counted() -> None:
    monitor = ModelDecisionMonitor(
        threshold_margin=0.005
    )

    monitor.record(
        calibrated_probability=0.048,
        decision="REVIEW",
    )

    snapshot = monitor.snapshot()

    assert (
        snapshot.near_review_threshold_total
        == 1
    )
    assert (
        snapshot.near_review_threshold_rate
        == 1.0
    )


def test_block_threshold_proximity_is_counted() -> None:
    monitor = ModelDecisionMonitor(
        threshold_margin=0.005
    )

    monitor.record(
        calibrated_probability=0.10,
        decision="BLOCK",
    )

    snapshot = monitor.snapshot()

    assert (
        snapshot.near_block_threshold_total
        == 1
    )
    assert (
        snapshot.near_block_threshold_rate
        == 1.0
    )


def test_probability_outside_margin_not_counted() -> None:
    monitor = ModelDecisionMonitor(
        threshold_margin=0.001
    )

    monitor.record(
        calibrated_probability=0.06,
        decision="REVIEW",
    )

    snapshot = monitor.snapshot()

    assert (
        snapshot.near_review_threshold_total
        == 0
    )
    assert (
        snapshot.near_block_threshold_total
        == 0
    )


@pytest.mark.parametrize(
    "probability",
    [
        -0.01,
        1.01,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_invalid_probability_rejected(
    probability: float,
) -> None:
    monitor = ModelDecisionMonitor()

    with pytest.raises(ValueError):
        monitor.record(
            calibrated_probability=probability,
            decision="APPROVE",
        )


def test_unknown_decision_rejected() -> None:
    monitor = ModelDecisionMonitor()

    with pytest.raises(ValueError):
        monitor.record(
            calibrated_probability=0.05,
            decision="UNKNOWN",
        )


@pytest.mark.parametrize(
    (
        "review_threshold",
        "block_threshold",
        "margin",
    ),
    [
        (-0.1, 0.10, 0.005),
        (1.1, 1.2, 0.005),
        (0.05, -0.1, 0.005),
        (0.05, 1.1, 0.005),
        (0.10, 0.10, 0.005),
        (0.20, 0.10, 0.005),
        (0.05, 0.10, -0.001),
    ],
)
def test_invalid_monitor_configuration(
    review_threshold: float,
    block_threshold: float,
    margin: float,
) -> None:
    with pytest.raises(ValueError):
        ModelDecisionMonitor(
            review_threshold=review_threshold,
            block_threshold=block_threshold,
            threshold_margin=margin,
        )


def test_decision_is_case_insensitive() -> None:
    monitor = ModelDecisionMonitor()

    monitor.record(
        calibrated_probability=0.02,
        decision="approve",
    )

    assert monitor.snapshot().approve_total == 1