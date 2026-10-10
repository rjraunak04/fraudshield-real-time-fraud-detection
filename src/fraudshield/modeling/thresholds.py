"""Threshold and business-cost optimization for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudshield.modeling.metrics import evaluate_fraud_predictions


@dataclass(frozen=True)
class BusinessCostAssumptions:
    """Explicit illustrative operating-cost assumptions."""

    false_positive_review_cost: float = 50.0
    detected_fraud_review_cost: float = 10.0


def _validate_inputs(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    transaction_amounts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate threshold-optimization inputs."""
    target = np.asarray(y_true)
    scores = np.asarray(
        probabilities,
        dtype=float,
    )
    amounts = np.asarray(
        transaction_amounts,
        dtype=float,
    )

    if target.ndim != 1:
        raise ValueError(
            "Target must be one-dimensional."
        )

    if scores.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional."
        )

    if amounts.ndim != 1:
        raise ValueError(
            "Transaction amounts must be one-dimensional."
        )

    if len(target) == 0:
        raise ValueError(
            "Inputs must not be empty."
        )

    if not (
        len(target)
        == len(scores)
        == len(amounts)
    ):
        raise ValueError(
            "Target, probabilities, and amounts "
            "must have equal length."
        )

    if set(
        np.unique(target).tolist()
    ) != {0, 1}:
        raise ValueError(
            "Target must contain both classes."
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "Probabilities contain non-finite values."
        )

    if (
        (scores < 0.0)
        | (scores > 1.0)
    ).any():
        raise ValueError(
            "Probabilities must lie in [0, 1]."
        )

    if not np.isfinite(
        amounts
    ).all():
        raise ValueError(
            "Transaction amounts contain "
            "non-finite values."
        )

    if (
        amounts < 0.0
    ).any():
        raise ValueError(
            "Transaction amounts must be non-negative."
        )

    return (
        target.astype(int),
        scores,
        amounts,
    )


def business_cost(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    transaction_amounts: np.ndarray,
    *,
    threshold: float,
    assumptions: BusinessCostAssumptions,
) -> float:
    """Calculate illustrative operating cost at one threshold."""
    target, scores, amounts = (
        _validate_inputs(
            y_true,
            probabilities,
            transaction_amounts,
        )
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must lie in [0, 1]."
        )

    if (
        assumptions.false_positive_review_cost
        < 0.0
    ):
        raise ValueError(
            "False-positive review cost "
            "must be non-negative."
        )

    if (
        assumptions.detected_fraud_review_cost
        < 0.0
    ):
        raise ValueError(
            "Detected-fraud review cost "
            "must be non-negative."
        )

    predictions = (
        scores >= threshold
    ).astype(int)

    false_positive_mask = (
        (target == 0)
        & (predictions == 1)
    )

    false_negative_mask = (
        (target == 1)
        & (predictions == 0)
    )

    true_positive_mask = (
        (target == 1)
        & (predictions == 1)
    )

    false_positive_cost = float(
        false_positive_mask.sum()
        * assumptions.false_positive_review_cost
    )

    # Illustrative assumption:
    # missed fraud costs its full transaction amount.
    missed_fraud_cost = float(
        amounts[
            false_negative_mask
        ].sum()
    )

    detected_fraud_cost = float(
        true_positive_mask.sum()
        * assumptions.detected_fraud_review_cost
    )

    return (
        false_positive_cost
        + missed_fraud_cost
        + detected_fraud_cost
    )


def build_threshold_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    transaction_amounts: np.ndarray,
    *,
    thresholds: np.ndarray,
    assumptions: BusinessCostAssumptions,
) -> pd.DataFrame:
    """Evaluate fraud metrics and cost across thresholds."""
    target, scores, amounts = (
        _validate_inputs(
            y_true,
            probabilities,
            transaction_amounts,
        )
    )

    candidate_thresholds = np.asarray(
        thresholds,
        dtype=float,
    )

    if candidate_thresholds.ndim != 1:
        raise ValueError(
            "Thresholds must be one-dimensional."
        )

    if candidate_thresholds.size == 0:
        raise ValueError(
            "Threshold grid must not be empty."
        )

    if not np.isfinite(
        candidate_thresholds
    ).all():
        raise ValueError(
            "Threshold grid contains non-finite values."
        )

    if (
        (candidate_thresholds < 0.0)
        | (candidate_thresholds > 1.0)
    ).any():
        raise ValueError(
            "Threshold grid values must lie in [0, 1]."
        )

    rows: list[
        dict[str, float | int]
    ] = []

    for threshold in candidate_thresholds:
        threshold_value = float(
            threshold
        )

        metrics = (
            evaluate_fraud_predictions(
                target,
                scores,
                threshold=threshold_value,
                transaction_amounts=amounts,
            )
        )

        cost = business_cost(
            target,
            scores,
            amounts,
            threshold=threshold_value,
            assumptions=assumptions,
        )

        rows.append(
            {
                "threshold": threshold_value,
                "precision": (
                    metrics.precision
                ),
                "recall": (
                    metrics.recall
                ),
                "f1": (
                    metrics.f1
                ),
                "false_positive_rate": (
                    metrics.false_positive_rate
                ),
                "predicted_positive_rate": (
                    metrics.predicted_positive_rate
                ),
                "fraud_transaction_capture_rate": (
                    metrics.fraud_transaction_capture_rate
                ),
                "fraud_amount_capture_rate": (
                    metrics.fraud_amount_capture_rate
                ),
                "true_negatives": (
                    metrics.true_negatives
                ),
                "false_positives": (
                    metrics.false_positives
                ),
                "false_negatives": (
                    metrics.false_negatives
                ),
                "true_positives": (
                    metrics.true_positives
                ),
                "business_cost": (
                    cost
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def select_cost_optimal_threshold(
    threshold_table: pd.DataFrame,
) -> pd.Series:
    """Select the minimum-cost threshold deterministically."""
    if threshold_table.empty:
        raise ValueError(
            "Threshold table must not be empty."
        )

    required_columns = {
        "threshold",
        "business_cost",
        "recall",
        "false_positive_rate",
    }

    missing = (
        required_columns
        - set(
            threshold_table.columns
        )
    )

    if missing:
        raise ValueError(
            "Threshold table missing columns: "
            f"{sorted(missing)}"
        )

    ordered = (
        threshold_table.sort_values(
            by=[
                "business_cost",
                "recall",
                "false_positive_rate",
                "threshold",
            ],
            ascending=[
                True,
                False,
                True,
                False,
            ],
        )
    )

    return ordered.iloc[0]