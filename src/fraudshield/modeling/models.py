"""Model definitions for FraudShield."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression(
    *,
    class_weight: str | dict[int, float] | None = None,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Build the FraudShield logistic-regression baseline."""
    if max_iter <= 0:
        raise ValueError(
            "max_iter must be positive."
        )

    classifier = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )