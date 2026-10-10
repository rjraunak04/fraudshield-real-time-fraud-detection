"""Verify the FraudShield production scoring contract."""

from __future__ import annotations

from fraudshield.data.load import load_transactions
from fraudshield.modeling.calibration import fit_sigmoid_calibrator
from fraudshield.modeling.dataset import (
    ModelingPartition,
    build_modeling_dataset,
)
from fraudshield.modeling.imbalance import balanced_sample_weights
from fraudshield.modeling.models import build_hist_gradient_boosting
from fraudshield.modeling.train import (
    fit_classifier,
    predict_fraud_probabilities,
)
from fraudshield.scoring.service import (
    FROZEN_THRESHOLD,
    FraudScorer,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def slice_partition(
    partition: ModelingPartition,
    start: int,
    stop: int,
) -> ModelingPartition:
    """Slice a modeling partition chronologically."""
    return ModelingPartition(
        X=partition.X.iloc[start:stop].reset_index(drop=True),
        y=partition.y.iloc[start:stop].reset_index(drop=True),
        metadata=partition.metadata.iloc[start:stop].reset_index(drop=True),
    )


def main() -> None:
    """Verify the scorer against the frozen Day-03 pipeline."""
    transactions = load_transactions(DATASET_PATH)
    dataset = build_modeling_dataset(transactions)

    split_index = len(dataset.validation.X) // 2

    calibration_partition = slice_partition(
        dataset.validation,
        0,
        split_index,
    )

    weights = balanced_sample_weights(dataset.train.y)

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

    calibration_raw = predict_fraud_probabilities(
        trained,
        calibration_partition,
    )

    calibrator = fit_sigmoid_calibrator(
        calibration_partition.y.to_numpy(),
        calibration_raw,
    )

    scorer = FraudScorer(
        model=trained.model,
        calibrator=calibrator,
        threshold=FROZEN_THRESHOLD,
    )

    # Use a row from the validation threshold-selection window.
    # The locked test is intentionally not touched by CP-01.
    verification_index = split_index

    features = dataset.validation.X.iloc[verification_index]

    score = scorer.score_one(features)

    print()
    print(
        "FraudShield Day-04 CP-01 "
        "Production Scoring Contract"
    )
    print("=" * 100)

    print(f"Model version: {score.model_version}")
    print(f"Calibration: {score.calibration_method}")
    print(f"Feature count: {len(features)}")
    print(f"Raw probability: {score.raw_probability:.6f}")
    print(
        "Calibrated probability: "
        f"{score.calibrated_probability:.6f}"
    )
    print(f"Frozen threshold: {score.threshold:.6f}")
    print(f"High-risk flag: {score.is_high_risk}")
    print("Locked test used: False")

    if len(features) != 21:
        raise RuntimeError(
            "Production scorer did not receive "
            "the frozen 21-feature schema."
        )

    if score.threshold != FROZEN_THRESHOLD:
        raise RuntimeError(
            "Production threshold drift detected."
        )

    print()
    print(
        "CP-01 production scoring contract QA: PASSED"
    )


if __name__ == "__main__":
    main()