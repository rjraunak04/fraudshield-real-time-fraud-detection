"""Tests for FraudShield fraud-specific metrics."""

import numpy as np
import pandas as pd
import pytest

from fraudshield.modeling.metrics import (
    evaluate_fraud_predictions,
    metrics_to_frame,
)


def test_perfect_predictions_have_perfect_metrics() -> None:
    """Perfect probabilities should produce perfect classification."""
    target = np.array(
        [0, 0, 1, 1]
    )

    scores = np.array(
        [0.01, 0.10, 0.90, 0.99]
    )

    metrics = evaluate_fraud_predictions(
        target,
        scores,
    )

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.false_positive_rate == pytest.approx(0.0)


def test_confusion_matrix_counts_are_correct() -> None:
    """Thresholded predictions should produce correct counts."""
    target = np.array(
        [0, 0, 1, 1]
    )

    scores = np.array(
        [0.1, 0.8, 0.4, 0.9]
    )

    metrics = evaluate_fraud_predictions(
        target,
        scores,
        threshold=0.5,
    )

    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1


def test_fraud_transaction_capture_matches_recall() -> None:
    """Transaction capture rate should equal fraud recall."""
    target = np.array(
        [0, 1, 1, 1]
    )

    scores = np.array(
        [0.1, 0.9, 0.8, 0.2]
    )

    metrics = evaluate_fraud_predictions(
        target,
        scores,
        threshold=0.5,
    )

    assert (
        metrics.fraud_transaction_capture_rate
        == pytest.approx(2 / 3)
    )

    assert metrics.recall == pytest.approx(
        metrics.fraud_transaction_capture_rate
    )


def test_fraud_amount_capture_is_correct() -> None:
    """Detected fraud amount should be measured separately."""
    target = np.array(
        [0, 1, 1]
    )

    scores = np.array(
        [0.1, 0.9, 0.2]
    )

    amounts = np.array(
        [10.0, 80.0, 20.0]
    )

    metrics = evaluate_fraud_predictions(
        target,
        scores,
        threshold=0.5,
        transaction_amounts=amounts,
    )

    assert (
        metrics.fraud_amount_capture_rate
        == pytest.approx(0.8)
    )


def test_metrics_work_with_pandas_series() -> None:
    """Metric API should accept pandas Series."""
    target = pd.Series(
        [0, 0, 1, 1]
    )

    scores = pd.Series(
        [0.1, 0.2, 0.8, 0.9]
    )

    metrics = evaluate_fraud_predictions(
        target,
        scores,
    )

    assert metrics.true_positives == 2


def test_invalid_probability_is_rejected() -> None:
    """Probabilities outside zero-one range must fail."""
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        evaluate_fraud_predictions(
            np.array([0, 1]),
            np.array([0.2, 1.2]),
        )


def test_mismatched_score_length_is_rejected() -> None:
    """Targets and scores must have equal lengths."""
    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        evaluate_fraud_predictions(
            np.array([0, 1]),
            np.array([0.2]),
        )


def test_invalid_threshold_is_rejected() -> None:
    """Decision threshold must remain a probability."""
    with pytest.raises(
        ValueError,
        match="threshold",
    ):
        evaluate_fraud_predictions(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            threshold=1.1,
        )


def test_negative_transaction_amount_is_rejected() -> None:
    """Fraud amount metrics should reject negative amounts."""
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        evaluate_fraud_predictions(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            transaction_amounts=np.array(
                [10.0, -1.0]
            ),
        )


def test_metrics_to_frame_creates_model_row() -> None:
    """Metrics should convert to a comparison-friendly table."""
    metrics = evaluate_fraud_predictions(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )

    frame = metrics_to_frame(
        metrics,
        model_name="perfect",
    )

    assert len(frame) == 1
    assert frame.loc[0, "model"] == "perfect"
    assert frame.loc[0, "pr_auc"] == pytest.approx(1.0)