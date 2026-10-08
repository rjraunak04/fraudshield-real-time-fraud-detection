"""Evaluate global feature importance without reopening the locked test."""

from __future__ import annotations

from fraudshield.data.load import load_transactions
from fraudshield.explainability.global_importance import (
    compute_permutation_importance,
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
from fraudshield.modeling.train import fit_classifier

DATASET_PATH = "data/raw/fraudshield_transactions.csv"
EXPLANATION_ROWS = 10_000
N_REPEATS = 3


def slice_partition(
    partition: ModelingPartition,
    start: int,
    stop: int,
) -> ModelingPartition:
    """Slice a modeling partition chronologically."""
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
    """Run validation-only global explainability QA."""
    transactions = load_transactions(
        DATASET_PATH
    )
    dataset = build_modeling_dataset(
        transactions
    )

    weights = balanced_sample_weights(
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
        sample_weight=weights,
    )

    # Explain only the later validation window.
    # Do not use the locked test.
    validation_split = (
        len(dataset.validation.X) // 2
    )

    explanation_partition = (
        slice_partition(
            dataset.validation,
            validation_split,
            len(dataset.validation.X),
        )
    )

    sample_rows = min(
        EXPLANATION_ROWS,
        len(explanation_partition.X),
    )

    # Deterministic evenly spaced validation sample.
    sample_indices = (
        explanation_partition.X.index[
            :: max(
                1,
                len(explanation_partition.X)
                // sample_rows,
            )
        ][:sample_rows]
    )

    X_explain = (
        explanation_partition.X.loc[
            sample_indices
        ].reset_index(drop=True)
    )

    y_explain = (
        explanation_partition.y.loc[
            sample_indices
        ].reset_index(drop=True)
    )

    importance = (
        compute_permutation_importance(
            trained.model,
            X_explain,
            y_explain,
            scoring="average_precision",
            n_repeats=N_REPEATS,
            random_state=42,
        )
    )

    print()
    print(
        "FraudShield Day-04 CP-02 "
        "Global Model Explainability"
    )
    print("=" * 100)

    print(
        f"Explanation rows: {len(X_explain):,}"
    )
    print(
        f"Feature count: {len(X_explain.columns)}"
    )
    print(
        "Scoring metric: average_precision"
    )
    print(
        f"Permutation repeats: {N_REPEATS}"
    )
    print(
        "Data source: later validation window"
    )
    print(
        "Locked test used: False"
    )

    print()
    print("Top 10 global features")
    print("-" * 100)

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    if len(importance) != 21:
        raise RuntimeError(
            "Expected importance values "
            "for all 21 model features."
        )

    if importance["feature"].nunique() != 21:
        raise RuntimeError(
            "Feature importance output "
            "contains duplicate features."
        )

    if importance[
        "importance_mean"
    ].isna().any():
        raise RuntimeError(
            "Feature importance contains "
            "missing values."
        )

    print()
    print(
        "CP-02 global explainability QA: PASSED"
    )


if __name__ == "__main__":
    main()