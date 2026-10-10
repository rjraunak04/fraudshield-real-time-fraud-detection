"""Tests for FraudShield global model explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from fraudshield.explainability.global_importance import (
    compute_permutation_importance,
    top_global_features,
)
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS


def make_dataset(
    rows: int = 200,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create deterministic data with the frozen schema."""
    rng = np.random.default_rng(42)

    data = {
        column: rng.normal(
            size=rows
        )
        for column in MODEL_FEATURE_COLUMNS
    }

    X = pd.DataFrame(data)

    signal = (
        2.5 * X[MODEL_FEATURE_COLUMNS[0]]
        + 0.25
        * X[MODEL_FEATURE_COLUMNS[1]]
    )

    y = pd.Series(
        (signal > 0.0).astype(int),
        name="TX_FRAUD",
    )

    return X, y


def fitted_model() -> tuple[
    HistGradientBoostingClassifier,
    pd.DataFrame,
    pd.Series,
]:
    """Fit a small deterministic test model."""
    X, y = make_dataset()

    model = HistGradientBoostingClassifier(
        max_iter=30,
        random_state=42,
    )
    model.fit(X, y)

    return model, X, y


def test_compute_permutation_importance_returns_all_features() -> None:
    model, X, y = fitted_model()

    result = compute_permutation_importance(
        model,
        X,
        y,
        n_repeats=2,
        random_state=42,
    )

    assert len(result) == len(
        MODEL_FEATURE_COLUMNS
    )

    assert set(result["feature"]) == set(
        MODEL_FEATURE_COLUMNS
    )

    assert list(result["rank"]) == list(
        range(
            1,
            len(MODEL_FEATURE_COLUMNS) + 1,
        )
    )


def test_signal_feature_ranks_near_top() -> None:
    model, X, y = fitted_model()

    result = compute_permutation_importance(
        model,
        X,
        y,
        n_repeats=3,
        random_state=42,
    )

    top_features = set(
        result.head(3)["feature"]
    )

    assert (
        MODEL_FEATURE_COLUMNS[0]
        in top_features
    )


def test_importance_is_deterministic() -> None:
    model, X, y = fitted_model()

    first = compute_permutation_importance(
        model,
        X,
        y,
        n_repeats=2,
        random_state=42,
    )

    second = compute_permutation_importance(
        model,
        X,
        y,
        n_repeats=2,
        random_state=42,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_top_global_features_returns_requested_count() -> None:
    model, X, y = fitted_model()

    importance = (
        compute_permutation_importance(
            model,
            X,
            y,
            n_repeats=2,
            random_state=42,
        )
    )

    top = top_global_features(
        importance,
        top_n=5,
    )

    assert len(top) == 5
    assert top[0].rank == 1


def test_rejects_wrong_feature_schema() -> None:
    model, X, y = fitted_model()

    X = X.drop(
        columns=[
            MODEL_FEATURE_COLUMNS[-1]
        ]
    )

    with pytest.raises(
        ValueError,
        match="frozen model feature schema",
    ):
        compute_permutation_importance(
            model,
            X,
            y,
        )


def test_rejects_length_mismatch() -> None:
    model, X, y = fitted_model()

    with pytest.raises(
        ValueError,
        match="lengths must match",
    ):
        compute_permutation_importance(
            model,
            X,
            y.iloc[:-1],
        )


def test_rejects_single_class_target() -> None:
    model, X, _ = fitted_model()

    y = pd.Series(
        np.zeros(
            len(X),
            dtype=int,
        )
    )

    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        compute_permutation_importance(
            model,
            X,
            y,
        )


def test_rejects_missing_values() -> None:
    model, X, y = fitted_model()

    X = X.copy()
    X.loc[
        0,
        MODEL_FEATURE_COLUMNS[0],
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        compute_permutation_importance(
            model,
            X,
            y,
        )


@pytest.mark.parametrize(
    "top_n",
    [
        0,
        -1,
    ],
)
def test_top_global_features_rejects_invalid_top_n(
    top_n: int,
) -> None:
    importance = pd.DataFrame(
        {
            "rank": [1],
            "feature": [
                MODEL_FEATURE_COLUMNS[0]
            ],
            "importance_mean": [0.1],
            "importance_std": [0.01],
        }
    )

    with pytest.raises(
        ValueError,
        match="top_n",
    ):
        top_global_features(
            importance,
            top_n=top_n,
        )