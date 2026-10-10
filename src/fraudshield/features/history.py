"""Leakage-safe historical behavioural features for FraudShield."""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORICAL_FEATURE_COLUMNS = [
    "CUSTOMER_HIST_TX_COUNT",
    "CUSTOMER_HIST_MEAN_AMOUNT",
    "CUSTOMER_HIST_MAX_AMOUNT",
    "CUSTOMER_AMOUNT_TO_HIST_MEAN",
    "TERMINAL_HIST_TX_COUNT",
    "TERMINAL_HIST_MEAN_AMOUNT",
    "TERMINAL_AMOUNT_TO_HIST_MEAN",
]


def _validate_history_input(
    transactions: pd.DataFrame,
) -> None:
    """Validate columns required for historical features."""
    if transactions.empty:
        raise ValueError(
            "Cannot create historical features for an empty dataset."
        )

    required = {
        "TRANSACTION_ID",
        "TX_DATETIME",
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_AMOUNT",
    }

    missing = required.difference(transactions.columns)

    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(
            "Missing columns required for historical features: "
            f"{columns}"
        )

    if transactions["TRANSACTION_ID"].duplicated().any():
        raise ValueError(
            "TRANSACTION_ID must be unique for historical features."
        )

    if transactions["CUSTOMER_ID"].isna().any():
        raise ValueError(
            "CUSTOMER_ID must not contain missing values."
        )

    if transactions["TERMINAL_ID"].isna().any():
        raise ValueError(
            "TERMINAL_ID must not contain missing values."
        )

    if transactions["TX_AMOUNT"].isna().any():
        raise ValueError(
            "TX_AMOUNT must not contain missing values."
        )

    if (transactions["TX_AMOUNT"] < 0).any():
        raise ValueError(
            "TX_AMOUNT must not contain negative values."
        )


def _add_entity_history(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Add past-only historical statistics for one entity type."""
    working = frame[
        [
            "__ROW_ID__",
            "TRANSACTION_ID",
            "TX_DATETIME",
            entity_column,
            "TX_AMOUNT",
        ]
    ].copy()

    working = working.sort_values(
        [
            entity_column,
            "TX_DATETIME",
            "TRANSACTION_ID",
        ]
    ).reset_index(drop=True)

    timestamp_groups = working.groupby(
        [
            entity_column,
            "TX_DATETIME",
        ],
        sort=False,
    )

    batch = timestamp_groups["TX_AMOUNT"].agg(
        BATCH_COUNT="size",
        BATCH_SUM="sum",
        BATCH_MAX="max",
    ).reset_index()

    entity_groups = batch.groupby(
        entity_column,
        sort=False,
    )

    batch["HIST_COUNT"] = (
        entity_groups["BATCH_COUNT"]
        .cumsum()
        .sub(batch["BATCH_COUNT"])
    )

    batch["HIST_SUM"] = (
        entity_groups["BATCH_SUM"]
        .cumsum()
        .sub(batch["BATCH_SUM"])
    )

    previous_batch_max = entity_groups[
        "BATCH_MAX"
    ].transform(
        lambda series: series.shift(1).cummax()
    )

    batch["HIST_MAX"] = previous_batch_max

    batch["HIST_MEAN"] = np.where(
        batch["HIST_COUNT"] > 0,
        batch["HIST_SUM"] / batch["HIST_COUNT"],
        0.0,
    )

    batch["HIST_MAX"] = batch[
        "HIST_MAX"
    ].fillna(0.0)

    history = working.merge(
        batch[
            [
                entity_column,
                "TX_DATETIME",
                "HIST_COUNT",
                "HIST_MEAN",
                "HIST_MAX",
            ]
        ],
        on=[
            entity_column,
            "TX_DATETIME",
        ],
        how="left",
        validate="many_to_one",
    )

    count_column = f"{prefix}_HIST_TX_COUNT"
    mean_column = f"{prefix}_HIST_MEAN_AMOUNT"

    history[count_column] = history[
        "HIST_COUNT"
    ].astype("int32")

    history[mean_column] = history[
        "HIST_MEAN"
    ].astype("float64")

    selected_columns = [
        "__ROW_ID__",
        count_column,
        mean_column,
    ]

    if prefix == "CUSTOMER":
        max_column = "CUSTOMER_HIST_MAX_AMOUNT"

        history[max_column] = history[
            "HIST_MAX"
        ].astype("float64")

        selected_columns.append(max_column)

    return history[selected_columns]


def _safe_amount_ratio(
    amount: pd.Series,
    historical_mean: pd.Series,
) -> pd.Series:
    """Calculate current amount relative to a non-zero past mean."""
    ratio = np.divide(
        amount.to_numpy(dtype="float64"),
        historical_mean.to_numpy(dtype="float64"),
        out=np.zeros(
            len(amount),
            dtype="float64",
        ),
        where=historical_mean.to_numpy(
            dtype="float64"
        )
        > 0,
    )

    return pd.Series(
        ratio,
        index=amount.index,
        dtype="float64",
    )


def add_historical_behaviour_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Add customer and terminal past-only behavioural features.

    Transactions sharing an entity and exact timestamp receive the same
    historical state. The current timestamp batch is added to history
    only for later timestamps.
    """
    _validate_history_input(transactions)

    frame = transactions.copy()

    frame["TX_DATETIME"] = pd.to_datetime(
        frame["TX_DATETIME"],
        errors="coerce",
    )

    if frame["TX_DATETIME"].isna().any():
        raise ValueError(
            "TX_DATETIME contains invalid datetime values."
        )

    frame["__ROW_ID__"] = np.arange(
        len(frame),
        dtype="int64",
    )

    customer_history = _add_entity_history(
        frame,
        entity_column="CUSTOMER_ID",
        prefix="CUSTOMER",
    )

    terminal_history = _add_entity_history(
        frame,
        entity_column="TERMINAL_ID",
        prefix="TERMINAL",
    )

    result = frame.merge(
        customer_history,
        on="__ROW_ID__",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        terminal_history,
        on="__ROW_ID__",
        how="left",
        validate="one_to_one",
    )

    result["CUSTOMER_AMOUNT_TO_HIST_MEAN"] = (
        _safe_amount_ratio(
            result["TX_AMOUNT"],
            result["CUSTOMER_HIST_MEAN_AMOUNT"],
        )
    )

    result["TERMINAL_AMOUNT_TO_HIST_MEAN"] = (
        _safe_amount_ratio(
            result["TX_AMOUNT"],
            result["TERMINAL_HIST_MEAN_AMOUNT"],
        )
    )

    result = (
        result.sort_values("__ROW_ID__")
        .drop(columns=["__ROW_ID__"])
        .reset_index(drop=True)
    )

    return result