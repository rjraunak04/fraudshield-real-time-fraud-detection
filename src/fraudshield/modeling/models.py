"""Model definitions for FraudShield."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
        raise ValueError("max_iter must be positive.")

    classifier = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def build_random_forest(
    *,
    n_estimators: int = 200,
    max_depth: int | None = 12,
    min_samples_leaf: int = 20,
    class_weight: str | dict[int, float] | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build the unweighted Random Forest baseline."""
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")

    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be positive or None.")

    if min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be positive.")

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            ("classifier", classifier),
        ]
    )


def build_hist_gradient_boosting(
    *,
    learning_rate: float = 0.1,
    max_iter: int = 150,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
    random_state: int = 42,
) -> Pipeline:
    """Build the HistGradientBoosting baseline."""
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if max_leaf_nodes < 2:
        raise ValueError("max_leaf_nodes must be at least 2.")

    if min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be positive.")

    if l2_regularization < 0.0:
        raise ValueError("l2_regularization must be non-negative.")

    classifier = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("classifier", classifier),
        ]
    )