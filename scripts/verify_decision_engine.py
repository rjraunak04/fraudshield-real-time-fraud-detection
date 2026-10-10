"""Verify the FraudShield three-way risk decision engine."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from fraudshield.data.load import load_transactions
from fraudshield.decision.engine import (
    DecisionAction,
    RiskDecisionEngine,
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
from fraudshield.scoring.service import (
    FraudScore,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def main() -> None:
    """Run validation-only decision-engine QA."""
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

    calibration_X = (
        dataset.validation.X.iloc[
            :split_index
        ]
    )
    calibration_y = (
        dataset.validation.y.iloc[
            :split_index
        ]
    )

    decision_X = (
        dataset.validation.X.iloc[
            split_index:
        ]
    )

    raw_calibration_probability = (
        trained.model.predict_proba(
            calibration_X
        )[:, 1]
    )

    calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=42,
    )

    calibrator.fit(
        raw_calibration_probability.reshape(
            -1,
            1,
        ),
        calibration_y,
    )

    raw_decision_probability = (
        trained.model.predict_proba(
            decision_X
        )[:, 1]
    )

    calibrated_probability = (
        calibrator.predict_proba(
            raw_decision_probability.reshape(
                -1,
                1,
            )
        )[:, 1]
    )

    engine = RiskDecisionEngine()

    counts = {
        DecisionAction.APPROVE: 0,
        DecisionAction.REVIEW: 0,
        DecisionAction.BLOCK: 0,
    }

    for raw_probability, probability in zip(
        raw_decision_probability,
        calibrated_probability,
        strict=True,
    ):
        score = FraudScore(
            raw_probability=float(
                raw_probability
            ),
            calibrated_probability=float(
                probability
            ),
            threshold=0.048,
            is_high_risk=(
                probability >= 0.048
            ),
            model_version=(
                "day-03-hgb-balanced-v1"
            ),
            calibration_method="sigmoid",
        )

        decision = engine.decide(score)

        counts[decision.action] += 1

    total = len(decision_X)

    print()
    print(
        "FraudShield Day-04 CP-05 "
        "Risk Decision Engine"
    )
    print("=" * 100)

    print(
        f"Decision rows: {total:,}"
    )
    print(
        "Data source: later validation window"
    )
    print(
        "Calibration source: earlier validation window"
    )
    print(
        "Locked test used: False"
    )

    print()
    print(
        "Decision policy"
    )
    print("-" * 100)
    print(
        f"Review threshold: "
        f"{engine.policy.review_threshold:.6f}"
    )
    print(
        f"Block threshold: "
        f"{engine.policy.block_threshold:.6f}"
    )
    print(
        f"Policy version: "
        f"{engine.policy.version}"
    )
    print(
        "Block threshold status: "
        "PROVISIONAL OPERATIONAL BOUNDARY"
    )

    print()
    print(
        "Decision distribution"
    )
    print("-" * 100)

    for action in DecisionAction:
        count = counts[action]
        rate = count / total

        print(
            f"{action.value:<8} "
            f"{count:>8,} "
            f"({rate:>8.4%})"
        )

    if sum(counts.values()) != total:
        raise RuntimeError(
            "Decision counts do not sum "
            "to the evaluation population."
        )

    if not np.isfinite(
        calibrated_probability
    ).all():
        raise RuntimeError(
            "Calibrated probabilities "
            "contain invalid values."
        )

    if (
        calibrated_probability.min() < 0.0
        or calibrated_probability.max() > 1.0
    ):
        raise RuntimeError(
            "Calibrated probabilities "
            "fall outside [0, 1]."
        )

    print()
    print(
        "CP-05 risk decision-engine QA: PASSED"
    )


if __name__ == "__main__":
    main()