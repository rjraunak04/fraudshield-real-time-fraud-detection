"""Tests for FraudShield imbalance utilities."""

import numpy as np
import pandas as pd
import pytest

from fraudshield.modeling.imbalance import (
    balanced_class_weights,
    balanced_sample_weights,
)


def test_balanced_class_weights_for_imbalanced_target() -> None:
    y = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    )

    weights = balanced_class_weights(y)

    assert weights[0] == pytest.approx(
        10 / 16
    )
    assert weights[1] == pytest.approx(
        10 / 4
    )


def test_balanced_class_weights_for_equal_classes() -> None:
    y = np.array(
        [0, 1, 0, 1]
    )

    weights = balanced_class_weights(y)

    assert weights == {
        0: 1.0,
        1: 1.0,
    }


def test_balanced_sample_weights_match_target() -> None:
    y = np.array(
        [0, 0, 0, 1]
    )

    weights = balanced_sample_weights(y)

    assert len(weights) == len(y)
    assert weights[3] > weights[0]


def test_balanced_sample_weights_accept_series() -> None:
    y = pd.Series(
        [0, 0, 1, 1]
    )

    weights = balanced_sample_weights(y)

    np.testing.assert_allclose(
        weights,
        np.ones(4),
    )


@pytest.mark.parametrize(
    "target",
    [
        np.array([]),
        np.array([0, 0, 0]),
        np.array([1, 1, 1]),
        np.array([0, 1, 2]),
    ],
)
def test_balanced_class_weights_reject_invalid_targets(
    target: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        balanced_class_weights(target)


def test_balanced_class_weights_reject_multidimensional_target() -> None:
    y = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        balanced_class_weights(y)