"""Verify FraudShield reason codes using real validation data."""

from __future__ import annotations

from fraudshield.data.load import load_transactions
from fraudshield.explainability.local import (
    build_reference_values,
    explain_transaction,
)
from fraudshield.explainability.reason_codes import (
    build_reason_codes,
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
    """Run real-data reason-code QA."""
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

    reference = build_reference_values(
        dataset.validation.X.iloc[
            :split_index
        ]
    )

    explanation_X = (
        dataset.validation.X.iloc[
            split_index:
        ]
    )

    explanation_y = (
        dataset.validation.y.iloc[
            split_index:
        ]
    )

    fraud_positions = (
        explanation_y.to_numpy()
        == 1
    ).nonzero()[0]

    if len(fraud_positions) == 0:
        raise RuntimeError(
            "No fraud transaction found in "
            "the validation explanation window."
        )

    position = int(
        fraud_positions[0]
    )

    local_explanation = explain_transaction(
        trained.model,
        explanation_X.iloc[position],
        reference,
        top_n=10,
    )

    reasons = build_reason_codes(
        local_explanation,
        max_reasons=3,
        risk_only=True,
    )

    print()
    print(
        "FraudShield Day-04 CP-04 "
        "Human-Readable Reason Codes"
    )
    print("=" * 100)

    print(
        "Data source: later validation window"
    )
    print(
        "Reference source: earlier validation window"
    )
    print("Actual label: 1")
    print("Locked test used: False")
    print(
        "Raw fraud probability: "
        f"{local_explanation.predicted_probability:.6f}"
    )

    print()
    print("Reviewer-facing risk reasons")
    print("-" * 100)

    if not reasons:
        print(
            "No increasing-risk reason was found "
            "among the selected local contributions."
        )
    else:
        for reason in reasons:
            print(
                f"{reason.rank:>2}. "
                f"{reason.code:<38} "
                f"delta={reason.probability_change:>+10.6f}"
            )
            print(
                f"    {reason.message}"
            )

    if len(reasons) > 3:
        raise RuntimeError(
            "Reason count exceeded the configured maximum."
        )

    if any(
        reason.direction != "increases_risk"
        for reason in reasons
    ):
        raise RuntimeError(
            "Risk-only output contains a non-risk reason."
        )

    if any(
        not reason.code
        or not reason.message
        for reason in reasons
    ):
        raise RuntimeError(
            "Reason code output is incomplete."
        )

    print()
    print(
        "CP-04 human-readable reason-code QA: PASSED"
    )


if __name__ == "__main__":
    main()