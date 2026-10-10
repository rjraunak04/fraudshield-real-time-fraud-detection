"""Evaluate FraudShield model selection and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from fraudshield.data.load import load_transactions
from fraudshield.modeling.calibration import (
    evaluate_calibration,
    fit_sigmoid_calibrator,
)
from fraudshield.modeling.dataset import (
    ModelingPartition,
    build_modeling_dataset,
)
from fraudshield.modeling.imbalance import (
    balanced_sample_weights,
)
from fraudshield.modeling.models import (
    build_hist_gradient_boosting,
)
from fraudshield.modeling.train import (
    fit_classifier,
    predict_fraud_probabilities,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def slice_partition(
    partition: ModelingPartition,
    start: int,
    stop: int,
) -> ModelingPartition:
    """Slice a modeling partition while preserving chronological order."""
    return ModelingPartition(
        X=partition.X.iloc[start:stop].reset_index(drop=True),
        y=partition.y.iloc[start:stop].reset_index(drop=True),
        metadata=partition.metadata.iloc[start:stop].reset_index(drop=True),
    )


def main() -> None:
    """Select and calibrate the CP-05 winning model."""
    transactions = load_transactions(
        DATASET_PATH
    )

    dataset = build_modeling_dataset(
        transactions
    )

    # Split the existing validation period chronologically:
    # first half -> calibration fitting
    # second half -> calibration evaluation
    split_index = len(
        dataset.validation.X
    ) // 2

    calibration_partition = slice_partition(
        dataset.validation,
        0,
        split_index,
    )

    evaluation_partition = slice_partition(
        dataset.validation,
        split_index,
        len(dataset.validation.X),
    )

    # Balanced sample weights are derived from training labels only.
    sample_weights = balanced_sample_weights(
        dataset.train.y
    )

    # CP-05 winner by validation PR-AUC.
    model = build_hist_gradient_boosting(
        learning_rate=0.1,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    # Fit the fraud classifier on TRAIN only.
    trained = fit_classifier(
        model,
        dataset.train,
        sample_weight=sample_weights,
    )

    # Raw probabilities for calibration-fitting window.
    calibration_raw = predict_fraud_probabilities(
        trained,
        calibration_partition,
    )

    # Raw probabilities for later calibration-evaluation window.
    evaluation_raw = predict_fraud_probabilities(
        trained,
        evaluation_partition,
    )

    # Fit sigmoid calibrator using only the first validation half.
    calibrator = fit_sigmoid_calibrator(
        calibration_partition.y.to_numpy(),
        calibration_raw,
    )

    # Apply the already-fitted calibrator to the later validation half.
    evaluation_calibrated = calibrator.predict(
        evaluation_raw
    )

    # Probability-quality metrics before calibration.
    raw_metrics = evaluate_calibration(
        evaluation_partition.y.to_numpy(),
        evaluation_raw,
    )

    # Probability-quality metrics after calibration.
    calibrated_metrics = evaluate_calibration(
        evaluation_partition.y.to_numpy(),
        evaluation_calibrated,
    )

    # Ranking metrics before calibration.
    raw_pr_auc = float(
        average_precision_score(
            evaluation_partition.y,
            evaluation_raw,
        )
    )

    raw_roc_auc = float(
        roc_auc_score(
            evaluation_partition.y,
            evaluation_raw,
        )
    )

    # Ranking metrics after calibration.
    calibrated_pr_auc = float(
        average_precision_score(
            evaluation_partition.y,
            evaluation_calibrated,
        )
    )

    calibrated_roc_auc = float(
        roc_auc_score(
            evaluation_partition.y,
            evaluation_calibrated,
        )
    )

    results = pd.DataFrame(
        [
            {
                "probabilities": "raw",
                "pr_auc": raw_pr_auc,
                "roc_auc": raw_roc_auc,
                "brier_score": raw_metrics.brier_score,
                "log_loss": raw_metrics.log_loss,
                "mean_probability": raw_metrics.mean_probability,
                "observed_prevalence": raw_metrics.observed_prevalence,
            },
            {
                "probabilities": "sigmoid_calibrated",
                "pr_auc": calibrated_pr_auc,
                "roc_auc": calibrated_roc_auc,
                "brier_score": calibrated_metrics.brier_score,
                "log_loss": calibrated_metrics.log_loss,
                "mean_probability": calibrated_metrics.mean_probability,
                "observed_prevalence": (
                    calibrated_metrics.observed_prevalence
                ),
            },
        ]
    )

    # Verify that calibration fitting happened before evaluation in time.
    calibration_end = calibration_partition.metadata[
        "TX_DATETIME"
    ].max()

    evaluation_start = evaluation_partition.metadata[
        "TX_DATETIME"
    ].min()

    chronological = bool(
        calibration_end <= evaluation_start
    )

    # Sigmoid calibration should preserve ranking because it is a
    # monotonic transformation of the original model score.
    raw_order = np.argsort(
        evaluation_raw,
        kind="stable",
    )

    calibrated_order = np.argsort(
        evaluation_calibrated,
        kind="stable",
    )

    rank_order_preserved = bool(
        np.array_equal(
            raw_order,
            calibrated_order,
        )
    )

    print()
    print(
        "FraudShield Day-03 CP-06 "
        "Model Selection + Probability Calibration"
    )
    print("=" * 120)

    print(
        "Selected model: "
        "hist_gradient_boosting_balanced"
    )

    print(
        "Selection basis: "
        "highest CP-05 validation PR-AUC"
    )

    print(
        f"Training rows: "
        f"{len(dataset.train.X)}"
    )

    print(
        f"Calibration rows: "
        f"{len(calibration_partition.X)}"
    )

    print(
        f"Calibration evaluation rows: "
        f"{len(evaluation_partition.X)}"
    )

    print(
        "Calibration method: sigmoid"
    )

    print(
        f"Chronological calibration split: "
        f"{chronological}"
    )

    print(
        "Locked test evaluated: False"
    )

    print()
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    brier_improvement = (
        raw_metrics.brier_score
        - calibrated_metrics.brier_score
    )

    log_loss_improvement = (
        raw_metrics.log_loss
        - calibrated_metrics.log_loss
    )

    print()
    print(
        "Brier improvement: "
        f"{brier_improvement:.6f}"
    )

    print(
        "Log-loss improvement: "
        f"{log_loss_improvement:.6f}"
    )

    print(
        "Rank order preserved: "
        f"{rank_order_preserved}"
    )

    if not chronological:
        raise RuntimeError(
            "Calibration split is not chronological."
        )

    if calibrated_metrics.brier_score > raw_metrics.brier_score:
        raise RuntimeError(
            "Calibration worsened Brier score."
        )

    if calibrated_metrics.log_loss > raw_metrics.log_loss:
        raise RuntimeError(
            "Calibration worsened log loss."
        )

    if not rank_order_preserved:
        raise RuntimeError(
            "Sigmoid calibration changed probability ranking."
        )

    print()
    print(
        "CP-06 probability calibration QA: PASSED"
    )


if __name__ == "__main__":
    main()