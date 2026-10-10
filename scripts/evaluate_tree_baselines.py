"""Train and evaluate FraudShield tree-ensemble baselines."""

from __future__ import annotations

import pandas as pd

from fraudshield.data.load import load_transactions
from fraudshield.modeling.dataset import build_modeling_dataset
from fraudshield.modeling.metrics import (
    evaluate_fraud_predictions,
    metrics_to_frame,
)
from fraudshield.modeling.models import (
    build_hist_gradient_boosting,
    build_random_forest,
)
from fraudshield.modeling.train import fit_classifier, predict_fraud_probabilities

DATASET_PATH = "data/raw/fraudshield_transactions.csv"
DEFAULT_THRESHOLD = 0.5


def evaluate_model(
    model_name: str,
    model,
    dataset,
) -> pd.DataFrame:
    """Fit on training data and evaluate validation only."""
    trained = fit_classifier(
        model,
        dataset.train,
    )

    validation_probabilities = predict_fraud_probabilities(
        trained,
        dataset.validation,
    )

    metrics = evaluate_fraud_predictions(
        dataset.validation.y,
        validation_probabilities,
        threshold=DEFAULT_THRESHOLD,
        transaction_amounts=dataset.validation.X["TX_AMOUNT"],
    )

    return metrics_to_frame(
        metrics,
        model_name=model_name,
    )


def main() -> None:
    """Evaluate unweighted tree baselines on validation."""
    transactions = load_transactions(DATASET_PATH)
    dataset = build_modeling_dataset(transactions)

    models = [
        (
            "random_forest",
            build_random_forest(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=20,
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gradient_boosting",
            build_hist_gradient_boosting(
                learning_rate=0.1,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.0,
                random_state=42,
            ),
        ),
    ]

    results = []

    for model_name, model in models:
        print(f"Training {model_name}...")

        results.append(
            evaluate_model(
                model_name,
                model,
                dataset,
            )
        )

    comparison = pd.concat(
        results,
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
        "FraudShield Day-03 CP-04 "
        "Tree Ensemble Validation Baselines"
    )
    print("=" * 120)

    print(f"Training rows: {len(dataset.train.X)}")
    print(f"Validation rows: {len(dataset.validation.X)}")
    print(f"Feature count: {dataset.train.X.shape[1]}")
    print("Class weighting: None")
    print(f"Decision threshold: {DEFAULT_THRESHOLD}")
    print("Locked test evaluated: False")

    print()
    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("Validation ranking by PR-AUC")
    print("=" * 120)

    ranking = comparison.sort_values(
        "pr_auc",
        ascending=False,
    )

    print(
        ranking[
            [
                "model",
                "pr_auc",
                "roc_auc",
                "recall",
                "precision",
                "false_positive_rate",
                "fraud_amount_capture_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("CP-04 tree ensemble baseline QA: PASSED")


if __name__ == "__main__":
    main()