"""Fraud-specific classification metrics for FraudShield."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class FraudMetrics:
    """Core fraud-classification evaluation metrics."""

    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    predicted_positive_rate: float
    fraud_transaction_capture_rate: float
    fraud_amount_capture_rate: float | None


def _validate_binary_target(
    y_true: pd.Series | np.ndarray,
) -> np.ndarray:
    """Validate and return a one-dimensional binary target."""
    values = np.asarray(y_true)

    if values.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "y_true must not be empty."
        )

    unique = set(np.unique(values).tolist())

    if unique != {0, 1}:
        raise ValueError(
            "y_true must contain both binary classes 0 and 1."
        )

    return values.astype(int)


def _validate_probabilities(
    y_score: pd.Series | np.ndarray,
    expected_length: int,
) -> np.ndarray:
    """Validate fraud probability scores."""
    scores = np.asarray(
        y_score,
        dtype=float,
    )

    if scores.ndim != 1:
        raise ValueError(
            "y_score must be one-dimensional."
        )

    if len(scores) != expected_length:
        raise ValueError(
            "y_true and y_score must have equal lengths."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "y_score contains non-finite values."
        )

    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError(
            "y_score must contain probabilities between 0 and 1."
        )

    return scores


def _validate_threshold(
    threshold: float,
) -> float:
    """Validate a classification threshold."""
    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    return threshold


def evaluate_fraud_predictions(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    *,
    threshold: float = 0.5,
    transaction_amounts: pd.Series | np.ndarray | None = None,
) -> FraudMetrics:
    """Evaluate fraud probabilities at a specified decision threshold."""
    target = _validate_binary_target(
        y_true
    )

    scores = _validate_probabilities(
        y_score,
        len(target),
    )

    threshold = _validate_threshold(
        threshold
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    ).ravel()

    negative_count = tn + fp

    false_positive_rate = (
        float(fp / negative_count)
        if negative_count
        else 0.0
    )

    fraud_count = tp + fn

    fraud_transaction_capture_rate = (
        float(tp / fraud_count)
        if fraud_count
        else 0.0
    )

    fraud_amount_capture_rate = None

    if transaction_amounts is not None:
        amounts = np.asarray(
            transaction_amounts,
            dtype=float,
        )

        if amounts.ndim != 1:
            raise ValueError(
                "transaction_amounts must be one-dimensional."
            )

        if len(amounts) != len(target):
            raise ValueError(
                "transaction_amounts must match y_true length."
            )

        if not np.isfinite(amounts).all():
            raise ValueError(
                "transaction_amounts contains non-finite values."
            )

        if (amounts < 0.0).any():
            raise ValueError(
                "transaction_amounts must be non-negative."
            )

        fraud_mask = target == 1
        detected_fraud_mask = (
            fraud_mask
            & (predictions == 1)
        )

        total_fraud_amount = float(
            amounts[fraud_mask].sum()
        )

        captured_fraud_amount = float(
            amounts[
                detected_fraud_mask
            ].sum()
        )

        fraud_amount_capture_rate = (
            captured_fraud_amount
            / total_fraud_amount
            if total_fraud_amount > 0.0
            else 0.0
        )

    return FraudMetrics(
        threshold=threshold,
        roc_auc=float(
            roc_auc_score(
                target,
                scores,
            )
        ),
        pr_auc=float(
            average_precision_score(
                target,
                scores,
            )
        ),
        precision=float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        false_positive_rate=false_positive_rate,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        predicted_positive_rate=float(
            predictions.mean()
        ),
        fraud_transaction_capture_rate=(
            fraud_transaction_capture_rate
        ),
        fraud_amount_capture_rate=(
            fraud_amount_capture_rate
        ),
    )


def metrics_to_frame(
    metrics: FraudMetrics,
    *,
    model_name: str,
) -> pd.DataFrame:
    """Convert a metrics result into a one-row table."""
    row = asdict(metrics)

    return pd.DataFrame(
        [
            {
                "model": model_name,
                **row,
            }
        ]
    )