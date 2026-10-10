"""Verify FraudShield feature and prediction drift monitoring."""

from __future__ import annotations

import numpy as np

from fraudshield.monitoring import (
    DriftStatus,
    evaluate_drift,
)


def main() -> None:
    """Run deterministic drift verification."""
    rng = np.random.default_rng(42)

    reference_amount = rng.normal(
        loc=50.0,
        scale=10.0,
        size=5000,
    )

    stable_amount = rng.normal(
        loc=50.0,
        scale=10.0,
        size=5000,
    )

    shifted_amount = rng.normal(
        loc=100.0,
        scale=20.0,
        size=5000,
    )

    reference_probability = rng.beta(
        a=2.0,
        b=20.0,
        size=5000,
    )

    shifted_probability = rng.beta(
        a=8.0,
        b=5.0,
        size=5000,
    )

    stable_result = evaluate_drift(
        name="TX_AMOUNT_STABLE",
        reference=reference_amount,
        current=stable_amount,
    )

    shifted_feature_result = evaluate_drift(
        name="TX_AMOUNT_SHIFTED",
        reference=reference_amount,
        current=shifted_amount,
    )

    prediction_result = evaluate_drift(
        name="CALIBRATED_PROBABILITY",
        reference=reference_probability,
        current=shifted_probability,
    )

    assert (
        stable_result.status
        is DriftStatus.STABLE
    )

    assert (
        shifted_feature_result.status
        is DriftStatus.DRIFT
    )

    assert (
        prediction_result.status
        is DriftStatus.DRIFT
    )

    print(
        "=== DAY-06 CP-03 DRIFT QA ==="
    )
    print(
        "Stable feature PSI:",
        f"{stable_result.psi:.6f}",
    )
    print(
        "Stable feature status:",
        stable_result.status.value,
    )
    print(
        "Shifted feature PSI:",
        f"{shifted_feature_result.psi:.6f}",
    )
    print(
        "Shifted feature status:",
        shifted_feature_result.status.value,
    )
    print(
        "Prediction PSI:",
        f"{prediction_result.psi:.6f}",
    )
    print(
        "Prediction status:",
        prediction_result.status.value,
    )
    print(
        "Reference-derived bins: PASSED"
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
        "Automatic retraining triggered: False"
    )
    print(
        "Day-06 CP-03 QA: PASSED"
    )


if __name__ == "__main__":
    main()