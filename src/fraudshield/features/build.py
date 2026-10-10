"""Unified feature engineering pipeline for FraudShield."""

from __future__ import annotations

import pandas as pd

from fraudshield.features.history import (
    HISTORICAL_FEATURE_COLUMNS,
    add_historical_behaviour_features,
)
from fraudshield.features.time import (
    TIME_FEATURE_COLUMNS,
    add_transaction_time_features,
)
from fraudshield.features.velocity import (
    VELOCITY_FEATURE_COLUMNS,
    add_customer_velocity_features,
)

MODEL_EXCLUDED_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
]


ENGINEERED_FEATURE_COLUMNS = [
    *TIME_FEATURE_COLUMNS,
    *VELOCITY_FEATURE_COLUMNS,
    *HISTORICAL_FEATURE_COLUMNS,
]


def build_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete leakage-safe FraudShield feature dataset."""
    if transactions.empty:
        raise ValueError(
            "Cannot build features for an empty transaction dataset."
        )

    featured = add_transaction_time_features(
        transactions
    )

    featured = add_customer_velocity_features(
        featured
    )

    featured = add_historical_behaviour_features(
        featured
    )

    return featured


def get_model_feature_columns(
    featured: pd.DataFrame,
) -> list[str]:
    """Return columns eligible for model input."""
    excluded = set(MODEL_EXCLUDED_COLUMNS)

    return [
        column
        for column in featured.columns
        if column not in excluded
    ]


def validate_feature_dataset(
    original: pd.DataFrame,
    featured: pd.DataFrame,
) -> None:
    """Validate final feature dataset integrity and leakage safeguards."""
    if len(original) != len(featured):
        raise ValueError(
            "Feature engineering changed the number of rows."
        )

    if not original["TRANSACTION_ID"].reset_index(
        drop=True
    ).equals(
        featured["TRANSACTION_ID"].reset_index(
            drop=True
        )
    ):
        raise ValueError(
            "Feature engineering changed transaction ordering."
        )

    missing_features = set(
        ENGINEERED_FEATURE_COLUMNS
    ).difference(featured.columns)

    if missing_features:
        columns = ", ".join(
            sorted(missing_features)
        )

        raise ValueError(
            f"Missing engineered features: {columns}"
        )

    missing_values = (
        featured[
            ENGINEERED_FEATURE_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_values:
        raise ValueError(
            "Engineered features contain missing values."
        )

    model_columns = get_model_feature_columns(
        featured
    )

    forbidden = {
        "TRANSACTION_ID",
        "TX_DATETIME",
        "TX_FRAUD",
        "TX_FRAUD_SCENARIO",
    }

    leaked = forbidden.intersection(
        model_columns
    )

    if leaked:
        columns = ", ".join(sorted(leaked))

        raise ValueError(
            f"Forbidden model columns detected: {columns}"
        )