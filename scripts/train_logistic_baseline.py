"""Train and evaluate the FraudShield logistic baseline."""

from __future__ import annotations

from fraudshield.data.load import load_transactions
from fraudshield.modeling.dataset import (
    build_modeling_dataset,
)
from fraudshield.modeling.metrics import (
    evaluate_fraud_predictions,
)
from fraudshield.modeling.models import (
    build_logistic_regression,
)
from fraudshield.modeling.train import (
    fit_classifier,
    logistic_coefficient_table,
    predict_fraud_probabilities,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"
DEFAULT_THRESHOLD = 0.5


def main() -> None:
    """Train on training data and evaluate validation only."""
    transactions = load_transactions(
        DATASET_PATH
    )

    dataset = build_modeling_dataset(
        transactions
    )

    model = build_logistic_regression(
        class_weight=None,
        max_iter=1000,
        random_state=42,
    )

    trained = fit_classifier(
        model,
        dataset.train,
    )

    validation_probabilities = (
        predict_fraud_probabilities(
            trained,
            dataset.validation,
        )
    )

    metrics = evaluate_fraud_predictions(
        dataset.validation.y,
        validation_probabilities,
        threshold=DEFAULT_THRESHOLD,
        transaction_amounts=dataset.validation.X[
            "TX_AMOUNT"
        ],
    )

    coefficients = (
        logistic_coefficient_table(
            trained
        )
    )

    print()
    print(
        "FraudShield Day-03 CP-03 "
        "Logistic Regression Baseline"
    )
    print("=" * 80)

    print(
        "Training rows:",
        trained.training_rows,
    )

    print(
        "Validation rows:",
        len(dataset.validation.X),
    )

    print(
        "Feature count:",
        len(
            trained.feature_columns
        ),
    )

    print(
        "Training fraud prevalence:",
        f"{trained.training_fraud_rate:.6f}",
    )

    print(
        "Class weighting:",
        None,
    )

    print(
        "Decision threshold:",
        DEFAULT_THRESHOLD,
    )

    print(
        "Locked test evaluated:",
        False,
    )

    print()
    print("Validation Metrics")
    print("=" * 80)

    print(
        "ROC-AUC:",
        f"{metrics.roc_auc:.6f}",
    )

    print(
        "PR-AUC:",
        f"{metrics.pr_auc:.6f}",
    )

    print(
        "Precision:",
        f"{metrics.precision:.6f}",
    )

    print(
        "Recall:",
        f"{metrics.recall:.6f}",
    )

    print(
        "F1:",
        f"{metrics.f1:.6f}",
    )

    print(
        "False positive rate:",
        f"{metrics.false_positive_rate:.6f}",
    )

    print(
        "Predicted positive rate:",
        f"{metrics.predicted_positive_rate:.6f}",
    )

    print(
        "Fraud transaction capture:",
        f"{metrics.fraud_transaction_capture_rate:.6f}",
    )

    if (
        metrics.fraud_amount_capture_rate
        is not None
    ):
        print(
            "Fraud amount capture:",
            f"{metrics.fraud_amount_capture_rate:.6f}",
        )

    print()
    print("Confusion Matrix Counts")
    print("=" * 80)

    print(
        "True negatives:",
        metrics.true_negatives,
    )

    print(
        "False positives:",
        metrics.false_positives,
    )

    print(
        "False negatives:",
        metrics.false_negatives,
    )

    print(
        "True positives:",
        metrics.true_positives,
    )

    print()
    print("Top Logistic Coefficients")
    print("=" * 80)

    print(
        coefficients.head(
            10
        ).to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "CP-03 logistic baseline QA: PASSED"
    )


if __name__ == "__main__":
    main()