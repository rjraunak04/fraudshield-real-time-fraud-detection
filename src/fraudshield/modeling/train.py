"""Training utilities for FraudShield classification models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fraudshield.modeling.dataset import ModelingPartition


@dataclass(frozen=True)
class TrainedClassifier:
    """A fitted classifier and its training metadata."""

    model: Pipeline
    feature_columns: tuple[str, ...]
    training_rows: int
    training_fraud_rate: float


def fit_classifier(
    model: Pipeline,
    train: ModelingPartition,
    *,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> TrainedClassifier:
    """Fit a classifier using the training partition only."""
    if train.X.empty:
        raise ValueError("Training feature matrix must not be empty.")

    if len(train.X) != len(train.y):
        raise ValueError(
            "Training features and target have different lengths."
        )

    if train.X.isna().any().any():
        raise ValueError("Training features contain missing values.")

    if set(train.y.unique()) != {0, 1}:
        raise ValueError(
            "Training target must contain both classes."
        )

    fit_kwargs: dict[str, object] = {}

    if sample_weight is not None:
        weights = np.asarray(
            sample_weight,
            dtype=float,
        )

        if weights.ndim != 1:
            raise ValueError(
                "sample_weight must be one-dimensional."
            )

        if len(weights) != len(train.y):
            raise ValueError(
                "sample_weight length must match training rows."
            )

        if not np.isfinite(weights).all():
            raise ValueError(
                "sample_weight contains non-finite values."
            )

        if (weights <= 0.0).any():
            raise ValueError(
                "sample_weight values must be positive."
            )

        fit_kwargs["classifier__sample_weight"] = weights

    model.fit(
        train.X,
        train.y,
        **fit_kwargs,
    )

    return TrainedClassifier(
        model=model,
        feature_columns=tuple(train.X.columns),
        training_rows=len(train.X),
        training_fraud_rate=float(train.y.mean()),
    )


def predict_fraud_probabilities(
    trained: TrainedClassifier,
    partition: ModelingPartition,
) -> np.ndarray:
    """Predict positive-class fraud probabilities."""
    expected_columns = list(trained.feature_columns)

    if partition.X.columns.tolist() != expected_columns:
        raise ValueError(
            "Prediction feature columns do not match "
            "the training feature contract."
        )

    probabilities = trained.model.predict_proba(
        partition.X
    )[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model produced non-finite probabilities."
        )

    if (
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ).any():
        raise ValueError(
            "Model produced invalid probabilities."
        )

    return probabilities


def logistic_coefficient_table(
    trained: TrainedClassifier,
) -> pd.DataFrame:
    """Return logistic-regression coefficients by feature."""
    classifier = trained.model.named_steps[
        "classifier"
    ]

    coefficients = np.asarray(
        classifier.coef_
    )

    if coefficients.shape != (
        1,
        len(trained.feature_columns),
    ):
        raise ValueError(
            "Unexpected logistic-regression coefficient shape."
        )

    table = pd.DataFrame(
        {
            "feature": trained.feature_columns,
            "coefficient": coefficients[0],
        }
    )

    table["abs_coefficient"] = (
        table["coefficient"].abs()
    )

    return (
        table.sort_values(
            "abs_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )