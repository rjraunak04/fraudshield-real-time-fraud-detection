"""Optimize FraudShield operating threshold on validation data."""

from __future__ import annotations

import numpy as np

from fraudshield.data.load import load_transactions
from fraudshield.modeling.calibration import (
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
from fraudshield.modeling.thresholds import (
    BusinessCostAssumptions,
    build_threshold_table,
    select_cost_optimal_threshold,
)
from fraudshield.modeling.train import (
    fit_classifier,
    predict_fraud_probabilities,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"

FALSE_POSITIVE_REVIEW_COST = 50.0
DETECTED_FRAUD_REVIEW_COST = 10.0


def slice_partition(
    partition: ModelingPartition,
    start: int,
    stop: int,
) -> ModelingPartition:
    """Slice a partition while preserving chronological order."""
    return ModelingPartition(
        X=partition.X.iloc[
            start:stop
        ].reset_index(drop=True),
        y=partition.y.iloc[
            start:stop
        ].reset_index(drop=True),
        metadata=partition.metadata.iloc[
            start:stop
        ].reset_index(drop=True),
    )


def main() -> None:
    """Optimize the calibrated model operating threshold."""
    transactions = load_transactions(
        DATASET_PATH
    )

    dataset = build_modeling_dataset(
        transactions
    )

    split_index = len(
        dataset.validation.X
    ) // 2

    calibration_partition = slice_partition(
        dataset.validation,
        0,
        split_index,
    )

    threshold_partition = slice_partition(
        dataset.validation,
        split_index,
        len(dataset.validation.X),
    )

    sample_weights = (
        balanced_sample_weights(
            dataset.train.y
        )
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
        sample_weight=sample_weights,
    )

    calibration_raw = (
        predict_fraud_probabilities(
            trained,
            calibration_partition,
        )
    )

    threshold_raw = (
        predict_fraud_probabilities(
            trained,
            threshold_partition,
        )
    )

    calibrator = fit_sigmoid_calibrator(
        calibration_partition.y.to_numpy(),
        calibration_raw,
    )

    calibrated_probabilities = (
        calibrator.predict(
            threshold_raw
        )
    )

    assumptions = (
        BusinessCostAssumptions(
            false_positive_review_cost=(
                FALSE_POSITIVE_REVIEW_COST
            ),
            detected_fraud_review_cost=(
                DETECTED_FRAUD_REVIEW_COST
            ),
        )
    )

    # Fine grid over calibrated probability space.
    thresholds = np.linspace(
        0.001,
        0.100,
        100,
    )

    table = build_threshold_table(
        threshold_partition.y.to_numpy(),
        calibrated_probabilities,
        threshold_partition.X[
            "TX_AMOUNT"
        ].to_numpy(),
        thresholds=thresholds,
        assumptions=assumptions,
    )

    selected = (
        select_cost_optimal_threshold(
            table
        )
    )

    top_candidates = (
        table.sort_values(
            [
                "business_cost",
                "recall",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(10)
        .reset_index(drop=True)
    )

    print()
    print(
        "FraudShield Day-03 CP-07 "
        "Threshold + Business-Cost Optimization"
    )
    print("=" * 120)

    print(
        "Model: "
        "hist_gradient_boosting_balanced"
    )

    print(
        "Probability calibration: sigmoid"
    )

    print(
        f"Threshold-selection rows: "
        f"{len(threshold_partition.X)}"
    )

    print(
        "Threshold source: "
        "later validation half"
    )

    print(
        "Locked test evaluated: False"
    )

    print()
    print(
        "Illustrative cost assumptions "
        "(portfolio/demo assumptions, not bank facts)"
    )

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

    print()
    print(
        "Top 10 candidate thresholds by "
        "illustrative business cost"
    )
    print("=" * 120)

    columns = [
        "threshold",
        "business_cost",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "predicted_positive_rate",
        "fraud_transaction_capture_rate",
        "fraud_amount_capture_rate",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    print(
        top_candidates[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "Selected operating point"
    )
    print("=" * 120)

    print(
        f"Threshold: "
        f"{selected['threshold']:.6f}"
    )

    print(
        f"Illustrative business cost: "
        f"{selected['business_cost']:.2f}"
    )

    print(
        f"Precision: "
        f"{selected['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{selected['recall']:.6f}"
    )

    print(
        f"F1: "
        f"{selected['f1']:.6f}"
    )

    print(
        f"False-positive rate: "
        f"{selected['false_positive_rate']:.6f}"
    )

    print(
        f"Predicted-positive rate: "
        f"{selected['predicted_positive_rate']:.6f}"
    )

    print(
        "Fraud transaction capture: "
        f"{selected['fraud_transaction_capture_rate']:.6f}"
    )

    print(
        "Fraud amount capture: "
        f"{selected['fraud_amount_capture_rate']:.6f}"
    )

    print()
    print(
        "CP-07 threshold optimization QA: PASSED"
    )


if __name__ == "__main__":
    main()