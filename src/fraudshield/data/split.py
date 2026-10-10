"""Leakage-safe temporal dataset splitting for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Container for chronological train, validation, and test sets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
) -> None:
    """Validate temporal split fractions."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )


def _validate_split_input(transactions: pd.DataFrame) -> None:
    """Validate minimum requirements for temporal splitting."""
    required = {
        "TRANSACTION_ID",
        "TX_DATETIME",
        "TX_FRAUD",
    }

    missing = required.difference(transactions.columns)

    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing columns required for temporal split: {columns}"
        )

    if transactions.empty:
        raise ValueError("Cannot split an empty transaction dataset.")

    if len(transactions) < 3:
        raise ValueError(
            "At least three transactions are required for splitting."
        )


def temporal_train_validation_test_split(
    transactions: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split transactions chronologically without random shuffling.

    The earliest observations are assigned to training, followed by
    validation and then the locked test partition.
    """
    _validate_split_input(transactions)

    _validate_split_fractions(
        train_fraction,
        validation_fraction,
    )

    ordered = (
        transactions.sort_values(
            ["TX_DATETIME", "TRANSACTION_ID"]
        )
        .reset_index(drop=True)
    )

    n_rows = len(ordered)

    train_end = int(n_rows * train_fraction)
    validation_end = int(
        n_rows * (train_fraction + validation_fraction)
    )

    if train_end == 0:
        raise ValueError("Training partition would be empty.")

    if validation_end <= train_end:
        raise ValueError("Validation partition would be empty.")

    if validation_end >= n_rows:
        raise ValueError("Test partition would be empty.")

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[
        train_end:validation_end
    ].copy()
    test = ordered.iloc[validation_end:].copy()

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )


def validate_temporal_split(
    split: TemporalSplit,
) -> None:
    """Validate ordering, overlap, and target integrity of a split."""
    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }

    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(
                f"{name} partition must not be empty."
            )

        if partition["TRANSACTION_ID"].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate transaction IDs."
            )

        if partition["TX_FRAUD"].nunique() < 2:
            raise ValueError(
                f"{name} must contain both target classes."
            )

    train_ids = set(split.train["TRANSACTION_ID"])
    validation_ids = set(
        split.validation["TRANSACTION_ID"]
    )
    test_ids = set(split.test["TRANSACTION_ID"])

    if train_ids & validation_ids:
        raise ValueError(
            "Train and validation partitions overlap."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Train and test partitions overlap."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation and test partitions overlap."
        )

    if (
        split.train["TX_DATETIME"].max()
        > split.validation["TX_DATETIME"].min()
    ):
        raise ValueError(
            "Training data extends beyond validation data."
        )

    if (
        split.validation["TX_DATETIME"].max()
        > split.test["TX_DATETIME"].min()
    ):
        raise ValueError(
            "Validation data extends beyond test data."
        )


def split_summary(
    split: TemporalSplit,
) -> pd.DataFrame:
    """Create a compact summary of temporal partitions."""
    rows = []

    for name, partition in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        rows.append(
            {
                "partition": name,
                "rows": len(partition),
                "start_datetime": partition[
                    "TX_DATETIME"
                ].min(),
                "end_datetime": partition[
                    "TX_DATETIME"
                ].max(),
                "fraud_count": int(
                    partition["TX_FRAUD"].sum()
                ),
                "fraud_rate": float(
                    partition["TX_FRAUD"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)