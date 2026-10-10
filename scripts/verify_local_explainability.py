"""Verify transaction-level explainability on real FraudShield data."""

from __future__ import annotations

from fraudshield.data.load import load_transactions
from fraudshield.explainability.local import (
    build_reference_values,
    explain_transaction,
)
from fraudshield.modeling.dataset import (
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


def main() -> None:
    """Run real-data local explainability QA."""
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

    split_index = (
        len(dataset.validation.X) // 2
    )

    reference_frame = (
        dataset.validation.X.iloc[
            :split_index
        ]
    )

    explanation_frame = (
        dataset.validation.X.iloc[
            split_index:
        ]
    )

    explanation_target = (
        dataset.validation.y.iloc[
            split_index:
        ]
    )

    reference = build_reference_values(
        reference_frame
    )

    # Prefer an actual fraud transaction from the
    # later validation window for an interpretable QA example.
    fraud_positions = (
        explanation_target.to_numpy()
        == 1
    ).nonzero()[0]

    if len(fraud_positions) == 0:
        raise RuntimeError(
            "No fraud transaction found in "
            "the explanation window."
        )

    position = int(
        fraud_positions[0]
    )

    transaction = (
        explanation_frame.iloc[position]
    )

    explanation = explain_transaction(
        trained.model,
        transaction,
        reference,
        top_n=5,
    )

    print()
    print(
        "FraudShield Day-04 CP-03 "
        "Transaction-Level Explainability"
    )
    print("=" * 100)

    print(
        "Data source: later validation window"
    )
    print(
        "Reference source: earlier validation window"
    )
    print(
        "Actual label: 1"
    )
    print(
        "Locked test used: False"
    )
    print(
        "Explanation method: "
        "reference-value perturbation"
    )

    print(
        "Predicted raw fraud probability: "
        f"{explanation.predicted_probability:.6f}"
    )

    print()
    print("Top local contributions")
    print("-" * 100)

    for contribution in (
        explanation.contributions
    ):
        print(
            f"{contribution.rank:>2}. "
            f"{contribution.feature:<32} "
            f"value={contribution.feature_value:>12.6f} "
            f"reference={contribution.reference_value:>12.6f} "
            f"delta={contribution.probability_change:>+12.6f} "
            f"{contribution.direction}"
        )

    if len(explanation.contributions) != 5:
        raise RuntimeError(
            "Expected five local contributions."
        )

    if not (
        0.0
        <= explanation.predicted_probability
        <= 1.0
    ):
        raise RuntimeError(
            "Invalid predicted probability."
        )

    print()
    print(
        "CP-03 transaction-level "
        "explainability QA: PASSED"
    )


if __name__ == "__main__":
    main()