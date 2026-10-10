"""Tests for FraudShield naive baselines."""

import numpy as np
import pytest

from fraudshield.modeling.baselines import (
    amount_rule_scores,
    constant_probability_baseline,
)


def test_constant_baseline_returns_requested_probability() -> None:
    """Constant baseline should assign identical probabilities."""
    scores = constant_probability_baseline(
        4,
        fraud_probability=0.02,
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [0.02, 0.02, 0.02, 0.02]
        ),
    )


def test_constant_baseline_rejects_invalid_probability() -> None:
    """Baseline probability must remain between zero and one."""
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        constant_probability_baseline(
            3,
            fraud_probability=1.2,
        )


def test_constant_baseline_rejects_empty_length() -> None:
    """Constant baseline requires at least one observation."""
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        constant_probability_baseline(
            0,
            fraud_probability=0.1,
        )


def test_amount_rule_scores_are_bounded() -> None:
    """Empirical amount scores must remain valid probabilities."""
    reference = np.array(
        [10.0, 20.0, 30.0, 40.0]
    )

    scores = amount_rule_scores(
        np.array(
            [5.0, 25.0, 50.0]
        ),
        reference_amounts=reference,
    )

    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_amount_rule_scores_increase_with_amount() -> None:
    """Larger transaction amounts should receive larger scores."""
    reference = np.array(
        [10.0, 20.0, 30.0, 40.0]
    )

    scores = amount_rule_scores(
        np.array(
            [5.0, 25.0, 50.0]
        ),
        reference_amounts=reference,
    )

    assert scores[0] < scores[1]
    assert scores[1] < scores[2]


def test_amount_rule_uses_reference_distribution() -> None:
    """Scoring should be determined by supplied training reference."""
    reference = np.array(
        [10.0, 20.0, 30.0, 40.0]
    )

    scores = amount_rule_scores(
        np.array(
            [20.0, 40.0]
        ),
        reference_amounts=reference,
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [0.5, 1.0]
        ),
    )


def test_amount_rule_rejects_negative_amounts() -> None:
    """Transaction amounts should never be negative."""
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        amount_rule_scores(
            np.array(
                [-1.0, 10.0]
            ),
            reference_amounts=np.array(
                [1.0, 2.0]
            ),
        )