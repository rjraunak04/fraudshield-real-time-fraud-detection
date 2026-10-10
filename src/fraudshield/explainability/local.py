"""Transaction-level model explanations for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS


@dataclass(frozen=True)
class LocalFeatureContribution:
    """Perturbation-based contribution for one feature."""

    feature: str
    feature_value: float
    reference_value: float
    probability_change: float
    direction: str
    rank: int


@dataclass(frozen=True)
class TransactionExplanation:
    """Stable transaction-level explanation contract."""

    predicted_probability: float
    contributions: tuple[LocalFeatureContribution, ...]


def explain_transaction(
    model: object,
    transaction: pd.Series | dict[str, float],
    reference_values: pd.Series | dict[str, float],
    *,
    top_n: int = 5,
) -> TransactionExplanation:
    """Explain one transaction using reference-value perturbations.

    For every feature, replace the observed value with a reference
    value while holding all other features fixed. The resulting
    probability change estimates how strongly that feature supports
    or suppresses the transaction's model score.

    This is a model-behaviour explanation, not a causal effect.
    """
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    row = _coerce_feature_row(
        transaction,
        name="transaction",
    )
    reference = _coerce_feature_row(
        reference_values,
        name="reference_values",
    )

    base_probability = _positive_probability(
        model,
        row.to_frame().T,
    )

    contributions: list[
        tuple[str, float, float, float]
    ] = []

    for feature in MODEL_FEATURE_COLUMNS:
        perturbed = row.copy()
        perturbed.loc[feature] = reference.loc[feature]

        perturbed_probability = _positive_probability(
            model,
            perturbed.to_frame().T,
        )

        probability_change = (
            base_probability - perturbed_probability
        )

        contributions.append(
            (
                feature,
                float(row.loc[feature]),
                float(reference.loc[feature]),
                float(probability_change),
            )
        )

    contributions.sort(
        key=lambda item: (
            -abs(item[3]),
            item[0],
        )
    )

    selected = contributions[
        : min(top_n, len(contributions))
    ]

    ranked = tuple(
        LocalFeatureContribution(
            feature=feature,
            feature_value=feature_value,
            reference_value=reference_value,
            probability_change=probability_change,
            direction=_direction(
                probability_change
            ),
            rank=index,
        )
        for index, (
            feature,
            feature_value,
            reference_value,
            probability_change,
        ) in enumerate(selected, start=1)
    )

    return TransactionExplanation(
        predicted_probability=base_probability,
        contributions=ranked,
    )


def build_reference_values(
    X_reference: pd.DataFrame,
) -> pd.Series:
    """Build deterministic median reference values."""
    _validate_feature_frame(X_reference)

    reference = X_reference.median(
        axis=0,
        numeric_only=True,
    )

    reference = reference.loc[
        MODEL_FEATURE_COLUMNS
    ].astype(float)

    if not np.isfinite(
        reference.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Reference values must be finite."
        )

    return reference


def _coerce_feature_row(
    values: pd.Series | dict[str, float],
    *,
    name: str,
) -> pd.Series:
    """Validate and order one model feature row."""
    if isinstance(values, dict):
        row = pd.Series(values)
    elif isinstance(values, pd.Series):
        row = values.copy()
    else:
        raise TypeError(
            f"{name} must be a pandas Series or dict."
        )

    expected = set(MODEL_FEATURE_COLUMNS)
    actual = set(row.index)

    missing = expected - actual
    extra = actual - expected

    if missing or extra:
        raise ValueError(
            f"{name} must match the frozen model "
            "feature schema exactly. "
            f"Missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    row = row.loc[
        MODEL_FEATURE_COLUMNS
    ]

    try:
        row = row.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric values."
        ) from exc

    if row.isna().any():
        raise ValueError(
            f"{name} must not contain missing values."
        )

    if not np.isfinite(
        row.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{name} must contain finite values."
        )

    return row


def _validate_feature_frame(
    X: pd.DataFrame,
) -> None:
    """Validate a reference feature matrix."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X_reference must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Reference feature matrix must not be empty."
        )

    if tuple(X.columns) != tuple(
        MODEL_FEATURE_COLUMNS
    ):
        raise ValueError(
            "Reference feature matrix must match the "
            "frozen model feature schema and order."
        )

    if X.isna().any().any():
        raise ValueError(
            "Reference feature matrix must not "
            "contain missing values."
        )

    try:
        values = X.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Reference feature matrix must contain "
            "numeric values."
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            "Reference feature matrix must contain "
            "finite values."
        )


def _positive_probability(
    model: object,
    X: pd.DataFrame,
) -> float:
    """Return validated positive-class probability."""
    predict_proba = getattr(
        model,
        "predict_proba",
        None,
    )

    if predict_proba is None:
        raise TypeError(
            "Model must implement predict_proba."
        )

    probabilities = np.asarray(
        predict_proba(X),
        dtype=float,
    )

    if probabilities.shape != (1, 2):
        raise ValueError(
            "Model predict_proba must return shape (1, 2)."
        )

    probability = float(
        probabilities[0, 1]
    )

    if (
        not np.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError(
            "Predicted probability must be finite "
            "and lie in [0, 1]."
        )

    return probability


def _direction(
    probability_change: float,
) -> str:
    """Translate signed contribution into direction."""
    tolerance = 1e-12

    if probability_change > tolerance:
        return "increases_risk"

    if probability_change < -tolerance:
        return "decreases_risk"

    return "neutral"