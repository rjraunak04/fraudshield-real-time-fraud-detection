"""Leakage-safe customer velocity features for FraudShield."""

from __future__ import annotations

from collections import deque

import pandas as pd

VELOCITY_FEATURE_COLUMNS = [
    "CUSTOMER_TX_COUNT_1H",
    "CUSTOMER_TX_AMOUNT_1H",
    "CUSTOMER_TX_COUNT_24H",
    "CUSTOMER_TX_AMOUNT_24H",
    "CUSTOMER_TX_COUNT_7D",
    "CUSTOMER_TX_AMOUNT_7D",
]


WINDOWS = {
    "1H": pd.Timedelta(hours=1),
    "24H": pd.Timedelta(hours=24),
    "7D": pd.Timedelta(days=7),
}


def _validate_velocity_input(
    transactions: pd.DataFrame,
) -> None:
    """Validate columns required for customer velocity features."""
    if transactions.empty:
        raise ValueError(
            "Cannot create velocity features for an empty dataset."
        )

    required = {
        "TRANSACTION_ID",
        "TX_DATETIME",
        "CUSTOMER_ID",
        "TX_AMOUNT",
    }

    missing = required.difference(transactions.columns)

    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing columns required for velocity features: {columns}"
        )

    if transactions["TRANSACTION_ID"].duplicated().any():
        raise ValueError(
            "TRANSACTION_ID must be unique for velocity features."
        )

    if transactions["CUSTOMER_ID"].isna().any():
        raise ValueError(
            "CUSTOMER_ID must not contain missing values."
        )

    if transactions["TX_AMOUNT"].isna().any():
        raise ValueError(
            "TX_AMOUNT must not contain missing values."
        )

    if (transactions["TX_AMOUNT"] < 0).any():
        raise ValueError(
            "TX_AMOUNT must not contain negative values."
        )


def add_customer_velocity_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Add past-only customer transaction velocity features.

    For each transaction, the engineered values describe customer
    activity strictly before the current transaction.

    Transactions sharing the exact same timestamp for a customer do not
    observe one another, preventing arbitrary transaction-ID ordering
    from becoming historical information.
    """
    _validate_velocity_input(transactions)

    frame = transactions.copy()

    frame["TX_DATETIME"] = pd.to_datetime(
        frame["TX_DATETIME"],
        errors="coerce",
    )

    if frame["TX_DATETIME"].isna().any():
        raise ValueError(
            "TX_DATETIME contains invalid datetime values."
        )

    original_order_column = "__ORIGINAL_ORDER__"

    frame[original_order_column] = range(len(frame))

    ordered = (
        frame.sort_values(
            [
                "CUSTOMER_ID",
                "TX_DATETIME",
                "TRANSACTION_ID",
            ]
        )
        .reset_index(drop=True)
    )

    for column in VELOCITY_FEATURE_COLUMNS:
        ordered[column] = 0.0

    for _, customer_index in ordered.groupby(
        "CUSTOMER_ID",
        sort=False,
    ).groups.items():
        indices = list(customer_index)

        histories = {
            name: deque()
            for name in WINDOWS
        }

        running_amounts = {
            name: 0.0
            for name in WINDOWS
        }

        position = 0

        while position < len(indices):
            current_index = indices[position]

            current_time = ordered.at[
                current_index,
                "TX_DATETIME",
            ]

            batch_indices: list[int] = []

            while position < len(indices):
                candidate_index = indices[position]

                candidate_time = ordered.at[
                    candidate_index,
                    "TX_DATETIME",
                ]

                if candidate_time != current_time:
                    break

                batch_indices.append(candidate_index)
                position += 1

            for window_name, window_size in WINDOWS.items():
                history = histories[window_name]

                cutoff = current_time - window_size

                while (
                    history
                    and history[0][0] < cutoff
                ):
                    _, expired_amount = history.popleft()

                    running_amounts[window_name] -= (
                        expired_amount
                    )

                count_column = (
                    f"CUSTOMER_TX_COUNT_{window_name}"
                )
                amount_column = (
                    f"CUSTOMER_TX_AMOUNT_{window_name}"
                )

                history_count = len(history)
                history_amount = running_amounts[
                    window_name
                ]

                for batch_index in batch_indices:
                    ordered.at[
                        batch_index,
                        count_column,
                    ] = history_count

                    ordered.at[
                        batch_index,
                        amount_column,
                    ] = history_amount

            for batch_index in batch_indices:
                amount = float(
                    ordered.at[
                        batch_index,
                        "TX_AMOUNT",
                    ]
                )

                for window_name in WINDOWS:
                    histories[window_name].append(
                        (
                            current_time,
                            amount,
                        )
                    )

                    running_amounts[window_name] += amount

    count_columns = [
        column
        for column in VELOCITY_FEATURE_COLUMNS
        if "_COUNT_" in column
    ]

    ordered[count_columns] = ordered[
        count_columns
    ].astype("int32")

    result = (
        ordered.sort_values(original_order_column)
        .drop(columns=[original_order_column])
        .reset_index(drop=True)
    )

    return result