"""Leakage-safe transaction-time features for FraudShield."""

from __future__ import annotations

import pandas as pd

TIME_FEATURE_COLUMNS = [
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
    "TX_DAY_OF_MONTH",
    "TX_MONTH",
    "TX_IS_WEEKEND",
    "TX_IS_NIGHT",
    "TX_IS_BUSINESS_HOURS",
]


def add_transaction_time_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Add features available from the transaction timestamp."""
    if transactions.empty:
        raise ValueError(
            "Cannot create time features for an empty dataset."
        )

    if "TX_DATETIME" not in transactions.columns:
        raise ValueError(
            "TX_DATETIME is required for transaction-time features."
        )

    frame = transactions.copy()

    timestamps = pd.to_datetime(
        frame["TX_DATETIME"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "TX_DATETIME contains invalid datetime values."
        )

    frame["TX_HOUR"] = timestamps.dt.hour.astype("int8")

    frame["TX_DAY_OF_WEEK"] = (
        timestamps.dt.dayofweek.astype("int8")
    )

    frame["TX_DAY_OF_MONTH"] = (
        timestamps.dt.day.astype("int8")
    )

    frame["TX_MONTH"] = (
        timestamps.dt.month.astype("int8")
    )

    frame["TX_IS_WEEKEND"] = (
        frame["TX_DAY_OF_WEEK"]
        .isin([5, 6])
        .astype("int8")
    )

    frame["TX_IS_NIGHT"] = (
        frame["TX_HOUR"]
        .between(0, 5)
        .astype("int8")
    )

    frame["TX_IS_BUSINESS_HOURS"] = (
        frame["TX_HOUR"]
        .between(9, 16)
        .astype("int8")
    )

    return frame