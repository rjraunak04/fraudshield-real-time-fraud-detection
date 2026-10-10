"""Global model explainability for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS


@dataclass(frozen=True)
class GlobalImportanceResult:
    """Stable global feature-importance result."""

    feature: str
    importance_mean: float
    importance_std: float
    rank: int


def compute_permutation_importance(
    model: object,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    scoring: str = "average_precision",
    n_repeats: int = 5,
    random_state: int = 42,
    max_samples: int | float = 1.0,
) -> pd.DataFrame:
    """Compute validation-only permutation importance."""
    _validate_inputs(
        X,
        y,
        n_repeats=n_repeats,
        max_samples=max_samples,
    )

    result = permutation_importance(
        model,
        X,
        np.asarray(y),
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        max_samples=max_samples,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance["absolute_importance"] = (
        importance["importance_mean"].abs()
    )

    importance = importance.sort_values(
        by=[
            "absolute_importance",
            "feature",
        ],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    importance["rank"] = (
        np.arange(len(importance)) + 1
    )

    return importance[
        [
            "rank",
            "feature",
            "importance_mean",
            "importance_std",
        ]
    ]


def top_global_features(
    importance: pd.DataFrame,
    *,
    top_n: int = 10,
) -> list[GlobalImportanceResult]:
    """Return the highest-ranked global features."""
    required_columns = {
        "rank",
        "feature",
        "importance_mean",
        "importance_std",
    }

    missing = required_columns - set(
        importance.columns
    )

    if missing:
        raise ValueError(
            "Importance table is missing columns: "
            f"{sorted(missing)}"
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    rows = importance.sort_values(
        "rank",
        kind="stable",
    ).head(top_n)

    return [
        GlobalImportanceResult(
            feature=str(row.feature),
            importance_mean=float(
                row.importance_mean
            ),
            importance_std=float(
                row.importance_std
            ),
            rank=int(row.rank),
        )
        for row in rows.itertuples(
            index=False
        )
    ]


def _validate_inputs(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int,
    max_samples: int | float,
) -> None:
    """Validate explainability inputs."""
    if X.empty:
        raise ValueError(
            "Feature matrix must not be empty."
        )

    if tuple(X.columns) != tuple(
        MODEL_FEATURE_COLUMNS
    ):
        raise ValueError(
            "Feature matrix must match the frozen "
            "model feature schema and order."
        )

    target = np.asarray(y)

    if target.ndim != 1:
        raise ValueError(
            "Target must be one-dimensional."
        )

    if len(X) != len(target):
        raise ValueError(
            "Feature and target lengths must match."
        )

    if len(np.unique(target)) < 2:
        raise ValueError(
            "Target must contain both classes."
        )

    if X.isna().any().any():
        raise ValueError(
            "Feature matrix must not contain missing values."
        )

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Feature matrix must contain finite values."
        )

    if n_repeats <= 0:
        raise ValueError(
            "n_repeats must be greater than zero."
        )

    if isinstance(max_samples, bool):
        raise ValueError(
            "max_samples must be an int or float."
        )

    if isinstance(max_samples, int):
        if max_samples <= 0:
            raise ValueError(
                "Integer max_samples must be positive."
            )
    elif isinstance(max_samples, float):
        if not 0.0 < max_samples <= 1.0:
            raise ValueError(
                "Float max_samples must lie in (0, 1]."
            )
    else:
        raise ValueError(
            "max_samples must be an int or float."
        )