"""Tests for FraudShield logistic-regression modeling."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudshield.modeling.dataset import ModelingPartition
from fraudshield.modeling.models import (
    build_logistic_regression,
)
from fraudshield.modeling.train import (
    fit_classifier,
    logistic_coefficient_table,
    predict_fraud_probabilities,
)


def make_partition(
    n_rows: int = 100,
) -> ModelingPartition:
    """Create a deterministic binary modeling partition."""
    index = np.arange(n_rows)

    feature_one = index.astype(float)

    feature_two = (
        index % 7
    ).astype(float)

    target = (
        index % 5 == 0
    ).astype(int)

    X = pd.DataFrame(
        {
            "feature_one": feature_one,
            "feature_two": feature_two,
        }
    )

    y = pd.Series(
        target,
        name="TX_FRAUD",
    )

    metadata = pd.DataFrame(
        {
            "TRANSACTION_ID": index,
            "TX_DATETIME": pd.date_range(
                "2018-04-01",
                periods=n_rows,
                freq="h",
            ),
        }
    )

    return ModelingPartition(
        X=X,
        y=y,
        metadata=metadata,
    )


def test_logistic_builder_returns_pipeline() -> None:
    """Logistic model should include preprocessing and classifier."""
    model = build_logistic_regression()

    assert isinstance(
        model,
        Pipeline,
    )

    assert list(
        model.named_steps
    ) == [
        "scaler",
        "classifier",
    ]

    assert isinstance(
        model.named_steps["scaler"],
        StandardScaler,
    )


def test_logistic_baseline_has_no_class_weight() -> None:
    """CP-03 baseline should not yet apply imbalance weighting."""
    model = build_logistic_regression()

    classifier = model.named_steps[
        "classifier"
    ]

    assert classifier.class_weight is None


def test_logistic_builder_accepts_class_weight() -> None:
    """Builder should support later imbalance experiments."""
    model = build_logistic_regression(
        class_weight="balanced"
    )

    assert (
        model.named_steps[
            "classifier"
        ].class_weight
        == "balanced"
    )


def test_logistic_builder_rejects_invalid_iterations() -> None:
    """Iteration limit must be positive."""
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        build_logistic_regression(
            max_iter=0
        )


def test_fit_classifier_records_training_metadata() -> None:
    """Training result should record reproducibility metadata."""
    partition = make_partition()

    trained = fit_classifier(
        build_logistic_regression(),
        partition,
    )

    assert trained.training_rows == 100
    assert trained.training_fraud_rate == pytest.approx(
        0.2
    )

    assert trained.feature_columns == (
        "feature_one",
        "feature_two",
    )


def test_scaler_is_fitted_from_training_partition() -> None:
    """Scaler statistics should equal training feature means."""
    partition = make_partition()

    trained = fit_classifier(
        build_logistic_regression(),
        partition,
    )

    scaler = trained.model.named_steps[
        "scaler"
    ]

    np.testing.assert_allclose(
        scaler.mean_,
        partition.X.mean().to_numpy(),
    )


def test_predict_probabilities_are_valid() -> None:
    """Predicted fraud probabilities should be bounded."""
    partition = make_partition()

    trained = fit_classifier(
        build_logistic_regression(),
        partition,
    )

    probabilities = predict_fraud_probabilities(
        trained,
        partition,
    )

    assert len(probabilities) == len(
        partition.X
    )

    assert np.isfinite(
        probabilities
    ).all()

    assert np.all(
        probabilities >= 0.0
    )

    assert np.all(
        probabilities <= 1.0
    )


def test_prediction_does_not_refit_scaler() -> None:
    """Validation prediction must not modify training statistics."""
    train = make_partition()

    trained = fit_classifier(
        build_logistic_regression(),
        train,
    )

    scaler = trained.model.named_steps[
        "scaler"
    ]

    original_mean = scaler.mean_.copy()

    validation = make_partition(
        50
    )

    validation_x = (
        validation.X + 10000.0
    )

    shifted_validation = ModelingPartition(
        X=validation_x,
        y=validation.y,
        metadata=validation.metadata,
    )

    predict_fraud_probabilities(
        trained,
        shifted_validation,
    )

    np.testing.assert_allclose(
        scaler.mean_,
        original_mean,
    )


def test_prediction_rejects_feature_contract_change() -> None:
    """Prediction features must match training columns exactly."""
    train = make_partition()

    trained = fit_classifier(
        build_logistic_regression(),
        train,
    )

    bad_x = train.X[
        [
            "feature_two",
            "feature_one",
        ]
    ]

    bad_partition = ModelingPartition(
        X=bad_x,
        y=train.y,
        metadata=train.metadata,
    )

    with pytest.raises(
        ValueError,
        match="feature columns",
    ):
        predict_fraud_probabilities(
            trained,
            bad_partition,
        )


def test_training_rejects_missing_values() -> None:
    """Training should reject incomplete feature matrices."""
    partition = make_partition()

    bad_x = partition.X.copy()

    bad_x.loc[
        0,
        "feature_one",
    ] = np.nan

    bad_partition = ModelingPartition(
        X=bad_x,
        y=partition.y,
        metadata=partition.metadata,
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        fit_classifier(
            build_logistic_regression(),
            bad_partition,
        )


def test_coefficient_table_contains_every_feature() -> None:
    """Coefficient audit should represent every training feature."""
    partition = make_partition()

    trained = fit_classifier(
        build_logistic_regression(),
        partition,
    )

    table = logistic_coefficient_table(
        trained
    )

    assert set(
        table["feature"]
    ) == set(
        partition.X.columns
    )

    assert {
        "feature",
        "coefficient",
        "abs_coefficient",
    }.issubset(
        table.columns
    )


def test_coefficient_table_sorted_by_magnitude() -> None:
    """Coefficient audit should rank absolute effects."""
    partition = make_partition()

    trained = fit_classifier(
        build_logistic_regression(),
        partition,
    )

    table = logistic_coefficient_table(
        trained
    )

    magnitudes = table[
        "abs_coefficient"
    ].to_numpy()

    assert np.all(
        magnitudes[:-1]
        >= magnitudes[1:]
    )
def test_fit_classifier_accepts_sample_weights() -> None:
    """Training should support positive per-row sample weights."""
    partition = make_partition()

    weights = np.where(
        partition.y.to_numpy() == 1,
        5.0,
        1.0,
    )

    trained = fit_classifier(
        build_logistic_regression(),
        partition,
        sample_weight=weights,
    )

    probabilities = predict_fraud_probabilities(
        trained,
        partition,
    )

    assert len(probabilities) == len(partition.X)


def test_fit_classifier_rejects_wrong_sample_weight_length() -> None:
    """Sample weights must align with training observations."""
    partition = make_partition()

    with pytest.raises(
        ValueError,
        match="length",
    ):
        fit_classifier(
            build_logistic_regression(),
            partition,
            sample_weight=np.ones(
                len(partition.y) - 1
            ),
        )


def test_fit_classifier_rejects_non_positive_sample_weights() -> None:
    """Sample weights must be strictly positive."""
    partition = make_partition()

    weights = np.ones(
        len(partition.y)
    )
    weights[0] = 0.0

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        fit_classifier(
            build_logistic_regression(),
            partition,
            sample_weight=weights,
        )


def test_fit_classifier_rejects_non_finite_sample_weights() -> None:
    """Sample weights must be finite."""
    partition = make_partition()

    weights = np.ones(
        len(partition.y)
    )
    weights[0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        fit_classifier(
            build_logistic_regression(),
            partition,
            sample_weight=weights,
        )