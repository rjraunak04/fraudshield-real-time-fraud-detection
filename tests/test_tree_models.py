"""Tests for FraudShield tree-ensemble model definitions."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from fraudshield.modeling.dataset import ModelingPartition
from fraudshield.modeling.models import (
    build_hist_gradient_boosting,
    build_random_forest,
)
from fraudshield.modeling.train import fit_classifier, predict_fraud_probabilities


def make_partition(n_rows: int = 120) -> ModelingPartition:
    """Create a deterministic binary modeling partition."""
    index = np.arange(n_rows)

    X = pd.DataFrame(
        {
            "feature_one": index.astype(float),
            "feature_two": (index % 9).astype(float),
            "feature_three": ((index * 3) % 11).astype(float),
        }
    )

    y = pd.Series(
        (index % 6 == 0).astype(int),
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


def test_random_forest_builder_returns_pipeline() -> None:
    model = build_random_forest()

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["classifier"]
    assert isinstance(
        model.named_steps["classifier"],
        RandomForestClassifier,
    )


def test_random_forest_baseline_is_unweighted() -> None:
    model = build_random_forest()

    assert model.named_steps["classifier"].class_weight is None


def test_random_forest_builder_accepts_class_weight() -> None:
    model = build_random_forest(class_weight="balanced")

    assert model.named_steps["classifier"].class_weight == "balanced"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_estimators": 0}, "n_estimators"),
        ({"max_depth": 0}, "max_depth"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
    ],
)
def test_random_forest_rejects_invalid_parameters(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_random_forest(**kwargs)


def test_hist_gradient_boosting_builder_returns_pipeline() -> None:
    model = build_hist_gradient_boosting()

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["classifier"]
    assert isinstance(
        model.named_steps["classifier"],
        HistGradientBoostingClassifier,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_leaf_nodes": 1}, "max_leaf_nodes"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"l2_regularization": -1.0}, "l2_regularization"),
    ],
)
def test_hist_gradient_boosting_rejects_invalid_parameters(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_hist_gradient_boosting(**kwargs)


@pytest.mark.parametrize(
    "builder",
    [
        build_random_forest,
        build_hist_gradient_boosting,
    ],
)
def test_tree_models_produce_valid_probabilities(builder) -> None:
    partition = make_partition()

    trained = fit_classifier(
        builder(),
        partition,
    )

    probabilities = predict_fraud_probabilities(
        trained,
        partition,
    )

    assert len(probabilities) == len(partition.X)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_random_forest_is_deterministic() -> None:
    partition = make_partition()

    first = fit_classifier(
        build_random_forest(
            n_estimators=20,
            n_jobs=1,
        ),
        partition,
    )

    second = fit_classifier(
        build_random_forest(
            n_estimators=20,
            n_jobs=1,
        ),
        partition,
    )

    first_probabilities = predict_fraud_probabilities(
        first,
        partition,
    )

    second_probabilities = predict_fraud_probabilities(
        second,
        partition,
    )

    np.testing.assert_allclose(
        first_probabilities,
        second_probabilities,
    )


def test_hist_gradient_boosting_is_deterministic() -> None:
    partition = make_partition()

    first = fit_classifier(
        build_hist_gradient_boosting(),
        partition,
    )

    second = fit_classifier(
        build_hist_gradient_boosting(),
        partition,
    )

    first_probabilities = predict_fraud_probabilities(
        first,
        partition,
    )

    second_probabilities = predict_fraud_probabilities(
        second,
        partition,
    )

    np.testing.assert_allclose(
        first_probabilities,
        second_probabilities,
    )