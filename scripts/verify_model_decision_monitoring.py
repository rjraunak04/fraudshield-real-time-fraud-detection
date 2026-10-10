"""Verify FraudShield model and decision monitoring."""

from __future__ import annotations

from fraudshield.decision import (
    FROZEN_REVIEW_THRESHOLD,
    PROVISIONAL_BLOCK_THRESHOLD,
)
from fraudshield.monitoring import (
    ModelDecisionMonitor,
)


def main() -> None:
    """Run deterministic model-output monitoring QA."""
    monitor = ModelDecisionMonitor()

    observations = [
        (0.020, "APPROVE"),
        (0.047, "APPROVE"),
        (0.048, "REVIEW"),
        (0.060, "REVIEW"),
        (0.099, "REVIEW"),
        (0.100, "BLOCK"),
        (0.120, "BLOCK"),
    ]

    for probability, decision in observations:
        monitor.record(
            calibrated_probability=probability,
            decision=decision,
        )

    snapshot = monitor.snapshot()

    assert snapshot.predictions_total == 7

    assert snapshot.approve_total == 2
    assert snapshot.review_total == 3
    assert snapshot.block_total == 2

    assert (
        snapshot.near_review_threshold_total
        == 2
    )
    assert (
        snapshot.near_block_threshold_total
        == 2
    )

    assert snapshot.probability_min == 0.020
    assert snapshot.probability_max == 0.120

    print(
        "=== DAY-06 CP-04 MODEL/DECISION QA ==="
    )
    print(
        "Predictions monitored:",
        snapshot.predictions_total,
    )
    print(
        "APPROVE:",
        snapshot.approve_total,
    )
    print(
        "REVIEW:",
        snapshot.review_total,
    )
    print(
        "BLOCK:",
        snapshot.block_total,
    )
    print(
        "High-risk rate:",
        f"{snapshot.high_risk_rate:.6f}",
    )
    print(
        "Mean calibrated probability:",
        f"{snapshot.mean_probability:.6f}",
    )
    print(
        "Near review threshold:",
        snapshot.near_review_threshold_total,
    )
    print(
        "Near block threshold:",
        snapshot.near_block_threshold_total,
    )
    print(
        "Frozen review threshold:",
        f"{FROZEN_REVIEW_THRESHOLD:.3f}",
    )
    print(
        "Provisional block threshold:",
        f"{PROVISIONAL_BLOCK_THRESHOLD:.3f}",
    )
    print(
        "Ground-truth performance metrics: "
        "NOT CLAIMED"
    )
    print(
        "Locked test used: False"
    )
    print(
        "Model retrained: False"
    )
    print(
        "Threshold retuned: False"
    )
    print(
        "Day-06 CP-04 QA: PASSED"
    )


if __name__ == "__main__":
    main()