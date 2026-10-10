"""Naive fraud-detection baselines for FraudShield."""

from __future__ import annotations

import numpy as np
import pandas as pd


def constant_probability_baseline(
    n_rows: int,
    *,
    fraud_probability: float,
) -> np.ndarray:
    """Return the same fraud probability for every transaction."""
    if n_rows <= 0:
        raise ValueError(
            "n_rows must be positive."
        )

    probability = float(
        fraud_probability
    )

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "fraud_probability must be between 0 and 1."
        )

    return np.full(
        n_rows,
        probability,
        dtype=float,
    )


def amount_rule_scores(
    amounts: pd.Series | np.ndarray,
    *,
    reference_amounts: pd.Series | np.ndarray,
) -> np.ndarray:
    """Create simple amount-based anomaly scores.

    The reference distribution should come from training data only.
    Scores are empirical percentiles in the training amount
    distribution and therefore remain between zero and one.
    """
    values = np.asarray(
        amounts,
        dtype=float,
    )

    reference = np.asarray(
        reference_amounts,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "amounts must be one-dimensional."
        )

    if reference.ndim != 1:
        raise ValueError(
            "reference_amounts must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "amounts must not be empty."
        )

    if len(reference) == 0:
        raise ValueError(
            "reference_amounts must not be empty."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "amounts contains non-finite values."
        )

    if not np.isfinite(reference).all():
        raise ValueError(
            "reference_amounts contains non-finite values."
        )

    if (values < 0.0).any():
        raise ValueError(
            "amounts must be non-negative."
        )

    if (reference < 0.0).any():
        raise ValueError(
            "reference_amounts must be non-negative."
        )

    sorted_reference = np.sort(
        reference
    )

    ranks = np.searchsorted(
        sorted_reference,
        values,
        side="right",
    )

    return (
        ranks.astype(float)
        / len(sorted_reference)
    )