"""Imbalance-handling utilities for FraudShield."""

from __future__ import annotations

import numpy as np
import pandas as pd


def balanced_class_weights(
    y: pd.Series | np.ndarray,
) -> dict[int, float]:
    """Compute inverse-frequency balanced binary class weights."""
    target = np.asarray(y)

    if target.ndim != 1:
        raise ValueError(
            "Target must be one-dimensional."
        )

    if target.size == 0:
        raise ValueError(
            "Target must not be empty."
        )

    classes = set(
        np.unique(target).tolist()
    )

    if classes != {0, 1}:
        raise ValueError(
            "Target must contain exactly classes 0 and 1."
        )

    negative_count = int(
        np.sum(target == 0)
    )
    positive_count = int(
        np.sum(target == 1)
    )
    total_count = target.size

    return {
        0: total_count / (
            2.0 * negative_count
        ),
        1: total_count / (
            2.0 * positive_count
        ),
    }


def balanced_sample_weights(
    y: pd.Series | np.ndarray,
) -> np.ndarray:
    """Create per-row balanced weights from the target."""
    target = np.asarray(y)

    weights = balanced_class_weights(
        target
    )

    return np.where(
        target == 1,
        weights[1],
        weights[0],
    ).astype(float)