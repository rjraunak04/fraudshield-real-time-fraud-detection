"""Run the final locked-test evaluation for FraudShield Day-03."""

from __future__ import annotations

import pandas as pd

from fraudshield.data.load import load_transactions
from fraudshield.modeling.calibration import (
    evaluate_calibration,
    fit_sigmoid_calibrator,
)
from fraudshield.modeling.dataset import (
    ModelingPartition,
    build_modeling_dataset,
)
from fraudshield.modeling.imbalance import balanced_sample_weights
from fraudshield.modeling.metrics import evaluate_fraud_predictions
from fraudshield.modeling.models import build_hist_gradient_boosting
from fraudshield.modeling.thresholds import (
    BusinessCostAssumptions,
    business_cost,
)
from fraudshield.modeling.train import (
    fit_classifier,
    predict_fraud_probabilities,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"

FINAL_THRESHOLD = 0.048

FALSE_POSITIVE_REVIEW_COST = 50.0
DETECTED_FRAUD_REVIEW_COST = 10.0


def slice_partition(
    partition: ModelingPartition,
    start: int,
    stop: int,
) -> ModelingPartition:
    """Slice a partition while preserving chronological order."""
    return ModelingPartition(
        X=partition.X.iloc[start:stop].reset_index(drop=True),
        y=partition.y.iloc[start:stop].reset_index(drop=True),
        metadata=partition.metadata.iloc[start:stop].reset_index(drop=True),
    )


def main() -> None:
    """Evaluate the frozen Day-03 pipeline on the locked test set."""
    transactions = load_transactions(
        DATASET_PATH
    )

    dataset = build_modeling_dataset(
        transactions
    )

    validation_split = (
        len(dataset.validation.X) // 2
    )

    calibration_partition = slice_partition(
        dataset.validation,
        0,
        validation_split,
    )

    threshold_partition = slice_partition(
        dataset.validation,
        validation_split,
        len(dataset.validation.X),
    )

    train_weights = balanced_sample_weights(
        dataset.train.y
    )

    model = build_hist_gradient_boosting(
        learning_rate=0.1,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    trained = fit_classifier(
        model,
        dataset.train,
        sample_weight=train_weights,
    )

    calibration_raw = (
        predict_fraud_probabilities(
            trained,
            calibration_partition,
        )
    )

    calibrator = fit_sigmoid_calibrator(
        calibration_partition.y.to_numpy(),
        calibration_raw,
    )

    threshold_raw = (
        predict_fraud_probabilities(
            trained,
            threshold_partition,
        )
    )

    threshold_calibrated = (
        calibrator.predict(
            threshold_raw
        )
    )

    validation_metrics = (
        evaluate_fraud_predictions(
            threshold_partition.y.to_numpy(),
            threshold_calibrated,
            threshold=FINAL_THRESHOLD,
            transaction_amounts=(
                threshold_partition.X[
                    "TX_AMOUNT"
                ].to_numpy()
            ),
        )
    )

    # The locked test is opened only here, after the complete
    # model/calibration/threshold configuration has been frozen.
    test_raw = predict_fraud_probabilities(
        trained,
        dataset.test,
    )

    test_calibrated = calibrator.predict(
        test_raw
    )

    test_metrics = evaluate_fraud_predictions(
        dataset.test.y.to_numpy(),
        test_calibrated,
        threshold=FINAL_THRESHOLD,
        transaction_amounts=(
            dataset.test.X[
                "TX_AMOUNT"
            ].to_numpy()
        ),
    )

    calibration_metrics = evaluate_calibration(
        dataset.test.y.to_numpy(),
        test_calibrated,
    )

    assumptions = BusinessCostAssumptions(
        false_positive_review_cost=(
            FALSE_POSITIVE_REVIEW_COST
        ),
        detected_fraud_review_cost=(
            DETECTED_FRAUD_REVIEW_COST
        ),
    )

    test_business_cost = business_cost(
        dataset.test.y.to_numpy(),
        test_calibrated,
        dataset.test.X[
            "TX_AMOUNT"
        ].to_numpy(),
        threshold=FINAL_THRESHOLD,
        assumptions=assumptions,
    )

    comparison = pd.DataFrame(
        [
            {
                "partition": (
                    "validation_threshold_window"
                ),
                "rows": len(
                    threshold_partition.X
                ),
                "fraud_rate": float(
                    threshold_partition.y.mean()
                ),
                "pr_auc": (
                    validation_metrics.pr_auc
                ),
                "roc_auc": (
                    validation_metrics.roc_auc
                ),
                "precision": (
                    validation_metrics.precision
                ),
                "recall": (
                    validation_metrics.recall
                ),
                "f1": (
                    validation_metrics.f1
                ),
                "false_positive_rate": (
                    validation_metrics.false_positive_rate
                ),
                "predicted_positive_rate": (
                    validation_metrics.predicted_positive_rate
                ),
                "fraud_tx_capture": (
                    validation_metrics.fraud_transaction_capture_rate
                ),
                "fraud_amount_capture": (
                    validation_metrics.fraud_amount_capture_rate
                ),
            },
            {
                "partition": "locked_test",
                "rows": len(
                    dataset.test.X
                ),
                "fraud_rate": float(
                    dataset.test.y.mean()
                ),
                "pr_auc": (
                    test_metrics.pr_auc
                ),
                "roc_auc": (
                    test_metrics.roc_auc
                ),
                "precision": (
                    test_metrics.precision
                ),
                "recall": (
                    test_metrics.recall
                ),
                "f1": (
                    test_metrics.f1
                ),
                "false_positive_rate": (
                    test_metrics.false_positive_rate
                ),
                "predicted_positive_rate": (
                    test_metrics.predicted_positive_rate
                ),
                "fraud_tx_capture": (
                    test_metrics.fraud_transaction_capture_rate
                ),
                "fraud_amount_capture": (
                    test_metrics.fraud_amount_capture_rate
                ),
            },
        ]
    )

    print()
    print(
        "FraudShield Day-03 CP-08 "
        "Final Locked-Test Evaluation"
    )
    print("=" * 120)

    print(
        "Frozen model: "
        "hist_gradient_boosting_balanced"
    )
    print(
        "Frozen calibration: sigmoid"
    )
    print(
        f"Frozen threshold: "
        f"{FINAL_THRESHOLD:.6f}"
    )
    print(
        "Threshold selection source: "
        "later validation half"
    )
    print(
        "Locked test used for tuning: False"
    )

    print()
    print(
        "Partition comparison"
    )
    print("=" * 120)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "Locked-test confusion matrix"
    )
    print("=" * 120)

    print(
        f"TN: {test_metrics.true_negatives}"
    )
    print(
        f"FP: {test_metrics.false_positives}"
    )
    print(
        f"FN: {test_metrics.false_negatives}"
    )
    print(
        f"TP: {test_metrics.true_positives}"
    )

    print()
    print(
        "Locked-test probability quality"
    )
    print("=" * 120)

    print(
        "Observed fraud prevalence: "
        f"{calibration_metrics.observed_prevalence:.6f}"
    )
    print(
        "Mean calibrated probability: "
        f"{calibration_metrics.mean_probability:.6f}"
    )
    print(
        "Brier score: "
        f"{calibration_metrics.brier_score:.6f}"
    )
    print(
        "Log loss: "
        f"{calibration_metrics.log_loss:.6f}"
    )

    print()
    print(
        "Illustrative locked-test business cost"
    )
    print("=" * 120)

    print(
        "False-positive review cost: "
        f"{FALSE_POSITIVE_REVIEW_COST:.2f}"
    )
    print(
        "Missed-fraud cost: "
        "actual transaction amount"
    )
    print(
        "Detected-fraud review cost: "
        f"{DETECTED_FRAUD_REVIEW_COST:.2f}"
    )
    print(
        "Total illustrative business cost: "
        f"{test_business_cost:.2f}"
    )

    if len(dataset.test.X) == 0:
        raise RuntimeError(
            "Locked test partition is empty."
        )

    if (
        dataset.test.metadata[
            "TX_DATETIME"
        ].min()
        <= threshold_partition.metadata[
            "TX_DATETIME"
        ].max()
    ):
        raise RuntimeError(
            "Locked test does not occur strictly "
            "after threshold-selection data."
        )

    print()
    print(
        "CP-08 locked-test evaluation QA: PASSED"
    )


if __name__ == "__main__":
    main()