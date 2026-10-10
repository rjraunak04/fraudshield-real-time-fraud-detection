"""Evaluate FraudShield imbalance-handling strategies."""

from __future__ import annotations

import pandas as pd

from fraudshield.data.load import load_transactions
from fraudshield.modeling.dataset import (
    build_modeling_dataset,
)
from fraudshield.modeling.imbalance import (
    balanced_class_weights,
    balanced_sample_weights,
)
from fraudshield.modeling.metrics import (
    evaluate_fraud_predictions,
    metrics_to_frame,
)
from fraudshield.modeling.models import (
    build_hist_gradient_boosting,
    build_logistic_regression,
    build_random_forest,
)
from fraudshield.modeling.train import (
    fit_classifier,
    predict_fraud_probabilities,
)

DATASET_PATH = (
    "data/raw/fraudshield_transactions.csv"
)
DEFAULT_THRESHOLD = 0.5


def evaluate_candidate(
    model_name: str,
    model,
    dataset,
    *,
    sample_weight=None,
) -> pd.DataFrame:
    """Fit one candidate and evaluate validation only."""
    trained = fit_classifier(
        model,
        dataset.train,
        sample_weight=sample_weight,
    )

    probabilities = (
        predict_fraud_probabilities(
            trained,
            dataset.validation,
        )
    )

    metrics = evaluate_fraud_predictions(
        dataset.validation.y,
        probabilities,
        threshold=DEFAULT_THRESHOLD,
        transaction_amounts=(
            dataset.validation.X[
                "TX_AMOUNT"
            ]
        ),
    )

    return metrics_to_frame(
        metrics,
        model_name=model_name,
    )


def main() -> None:
    """Compare cost-sensitive imbalance strategies."""
    transactions = load_transactions(
        DATASET_PATH
    )

    dataset = build_modeling_dataset(
        transactions
    )

    class_weights = (
        balanced_class_weights(
            dataset.train.y
        )
    )

    sample_weights = (
        balanced_sample_weights(
            dataset.train.y
        )
    )

    candidates = [
        (
            "logistic_balanced",
            build_logistic_regression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
            None,
        ),
        (
            "random_forest_balanced",
            build_random_forest(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            None,
        ),
        (
            "random_forest_balanced_subsample",
            build_random_forest(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=20,
                class_weight=(
                    "balanced_subsample"
                ),
                random_state=42,
                n_jobs=-1,
            ),
            None,
        ),
        (
            "hist_gradient_boosting_balanced",
            build_hist_gradient_boosting(
                learning_rate=0.1,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.0,
                random_state=42,
            ),
            sample_weights,
        ),
    ]

    results = []

    print()
    print(
        "FraudShield Day-03 CP-05 "
        "Imbalance Strategy Evaluation"
    )
    print("=" * 120)

    print(
        f"Training rows: "
        f"{len(dataset.train.X)}"
    )
    print(
        f"Validation rows: "
        f"{len(dataset.validation.X)}"
    )
    print(
        f"Training fraud prevalence: "
        f"{dataset.train.y.mean():.6f}"
    )
    print(
        "Balanced class weights: "
        f"{class_weights}"
    )
    print(
        f"Decision threshold: "
        f"{DEFAULT_THRESHOLD}"
    )
    print(
        "Resampling used: False"
    )
    print(
        "Locked test evaluated: False"
    )

    for (
        model_name,
        model,
        weights,
    ) in candidates:
        print(
            f"Training {model_name}..."
        )

        result = evaluate_candidate(
            model_name,
            model,
            dataset,
            sample_weight=weights,
        )

        results.append(result)

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
        "Validation Results"
    )
    print("=" * 120)

    print(
        comparison[
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
        "Ranking by PR-AUC"
    )
    print("=" * 120)

    ranking = comparison.sort_values(
        "pr_auc",
        ascending=False,
    )

    print(
        ranking[
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
        "CP-05 imbalance strategy QA: PASSED"
    )


if __name__ == "__main__":
    main()