"""Evaluate FraudShield naive baselines on validation data only."""

from __future__ import annotations

import pandas as pd

from fraudshield.data.load import load_transactions
from fraudshield.modeling.baselines import (
    amount_rule_scores,
    constant_probability_baseline,
)
from fraudshield.modeling.dataset import (
    build_modeling_dataset,
)
from fraudshield.modeling.metrics import (
    evaluate_fraud_predictions,
    metrics_to_frame,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"
DEFAULT_THRESHOLD = 0.5


def main() -> None:
    """Evaluate naive baselines without touching locked-test labels."""
    transactions = load_transactions(
        DATASET_PATH
    )

    dataset = build_modeling_dataset(
        transactions
    )

    train = dataset.train
    validation = dataset.validation

    training_fraud_rate = float(
        train.y.mean()
    )

    constant_scores = (
        constant_probability_baseline(
            len(validation.X),
            fraud_probability=training_fraud_rate,
        )
    )

    constant_metrics = evaluate_fraud_predictions(
        validation.y,
        constant_scores,
        threshold=DEFAULT_THRESHOLD,
        transaction_amounts=validation.X[
            "TX_AMOUNT"
        ],
    )

    amount_scores = amount_rule_scores(
        validation.X[
            "TX_AMOUNT"
        ],
        reference_amounts=train.X[
            "TX_AMOUNT"
        ],
    )

    amount_metrics = evaluate_fraud_predictions(
        validation.y,
        amount_scores,
        threshold=DEFAULT_THRESHOLD,
        transaction_amounts=validation.X[
            "TX_AMOUNT"
        ],
    )

    results = pd.concat(
        [
            metrics_to_frame(
                constant_metrics,
                model_name=(
                    "constant_training_prevalence"
                ),
            ),
            metrics_to_frame(
                amount_metrics,
                model_name="amount_percentile_rule",
            ),
        ],
        ignore_index=True,
    )

    display_columns = [
        "model",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "predicted_positive_rate",
        "fraud_transaction_capture_rate",
        "fraud_amount_capture_rate",
    ]

    print()
    print(
        "FraudShield Day-03 CP-02 "
        "Validation Baselines"
    )
    print("=" * 110)

    print(
        "Training fraud prevalence:",
        f"{training_fraud_rate:.6f}",
    )

    print(
        "Validation rows:",
        len(validation.X),
    )

    print(
        "Locked test evaluated:",
        False,
    )

    print()
    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "CP-02 baseline evaluation QA: PASSED"
    )


if __name__ == "__main__":
    main()