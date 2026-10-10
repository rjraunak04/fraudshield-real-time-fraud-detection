"""Leakage-safe modeling dataset construction for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraudshield.data.split import (
    TemporalSplit,
    temporal_train_validation_test_split,
    validate_temporal_split,
)
from fraudshield.features.build import (
    ENGINEERED_FEATURE_COLUMNS,
    build_features,
)

TARGET_COLUMN = "TX_FRAUD"

MODEL_FEATURE_COLUMNS = [
    "TX_AMOUNT",
    *ENGINEERED_FEATURE_COLUMNS,
]

FORBIDDEN_MODEL_COLUMNS = {
    "TRANSACTION_ID",
    "TX_DATETIME",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
}


@dataclass(frozen=True)
class ModelingPartition:
    """Feature matrix, target, and metadata for one partition."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class ModelingDataset:
    """Chronological train, validation, and locked-test datasets."""

    train: ModelingPartition
    validation: ModelingPartition
    test: ModelingPartition


def _validate_modeling_input(
    transactions: pd.DataFrame,
) -> None:
    """Validate requirements for constructing modeling datasets."""
    if transactions.empty:
        raise ValueError(
            "Cannot build a modeling dataset from empty transactions."
        )

    required = {
        "TRANSACTION_ID",
        "TX_DATETIME",
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_AMOUNT",
        TARGET_COLUMN,
    }

    missing = required.difference(
        transactions.columns
    )

    if missing:
        columns = ", ".join(sorted(missing))

        raise ValueError(
            "Missing columns required for modeling: "
            f"{columns}"
        )

    if transactions["TRANSACTION_ID"].duplicated().any():
        raise ValueError(
            "TRANSACTION_ID must be unique for modeling."
        )

    target_values = set(
        transactions[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "TX_FRAUD must contain only binary values 0 and 1."
        )

    if target_values != {0, 1}:
        raise ValueError(
            "TX_FRAUD must contain both target classes."
        )


def _build_partition(
    partition: pd.DataFrame,
) -> ModelingPartition:
    """Convert a transaction partition into model-ready components."""
    missing_features = set(
        MODEL_FEATURE_COLUMNS
    ).difference(partition.columns)

    if missing_features:
        columns = ", ".join(
            sorted(missing_features)
        )

        raise ValueError(
            f"Missing approved model features: {columns}"
        )

    X = partition[
        MODEL_FEATURE_COLUMNS
    ].copy()

    y = (
        partition[TARGET_COLUMN]
        .astype("int8")
        .copy()
    )

    metadata_columns = [
        column
        for column in (
            "TRANSACTION_ID",
            "TX_DATETIME",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_FRAUD_SCENARIO",
        )
        if column in partition.columns
    ]

    metadata = partition[
        metadata_columns
    ].copy()

    return ModelingPartition(
        X=X,
        y=y,
        metadata=metadata,
    )


def validate_modeling_dataset(
    dataset: ModelingDataset,
) -> None:
    """Validate model partitions and leakage safeguards."""
    expected_features = MODEL_FEATURE_COLUMNS

    for name, partition in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        if partition.X.empty:
            raise ValueError(
                f"{name} feature matrix must not be empty."
            )

        if len(partition.X) != len(partition.y):
            raise ValueError(
                f"{name} features and target have different lengths."
            )

        if partition.X.columns.tolist() != expected_features:
            raise ValueError(
                f"{name} does not match the model feature contract."
            )

        leaked = FORBIDDEN_MODEL_COLUMNS.intersection(
            partition.X.columns
        )

        if leaked:
            columns = ", ".join(
                sorted(leaked)
            )

            raise ValueError(
                f"{name} contains forbidden model columns: {columns}"
            )

        if partition.X.isna().any().any():
            raise ValueError(
                f"{name} model features contain missing values."
            )

        if set(partition.y.unique()) != {0, 1}:
            raise ValueError(
                f"{name} target must contain both classes."
            )

    train_ids = set(
        dataset.train.metadata["TRANSACTION_ID"]
    )

    validation_ids = set(
        dataset.validation.metadata["TRANSACTION_ID"]
    )

    test_ids = set(
        dataset.test.metadata["TRANSACTION_ID"]
    )

    if train_ids & validation_ids:
        raise ValueError(
            "Train and validation transactions overlap."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Train and test transactions overlap."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation and test transactions overlap."
        )

    train_end = dataset.train.metadata[
        "TX_DATETIME"
    ].max()

    validation_start = dataset.validation.metadata[
        "TX_DATETIME"
    ].min()

    validation_end = dataset.validation.metadata[
        "TX_DATETIME"
    ].max()

    test_start = dataset.test.metadata[
        "TX_DATETIME"
    ].min()

    if train_end > validation_start:
        raise ValueError(
            "Training observations extend beyond validation."
        )

    if validation_end > test_start:
        raise ValueError(
            "Validation observations extend beyond locked test."
        )


def build_modeling_dataset(
    transactions: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ModelingDataset:
    """Build features and chronological modeling partitions."""
    _validate_modeling_input(transactions)

    featured = build_features(
        transactions
    )

    split: TemporalSplit = (
        temporal_train_validation_test_split(
            featured,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
    )

    validate_temporal_split(split)

    dataset = ModelingDataset(
        train=_build_partition(split.train),
        validation=_build_partition(
            split.validation
        ),
        test=_build_partition(split.test),
    )

    validate_modeling_dataset(dataset)

    return dataset


def modeling_dataset_summary(
    dataset: ModelingDataset,
) -> pd.DataFrame:
    """Summarize chronological modeling partitions."""
    rows = []

    for name, partition in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        rows.append(
            {
                "partition": name,
                "rows": len(partition.X),
                "features": partition.X.shape[1],
                "fraud_count": int(
                    partition.y.sum()
                ),
                "fraud_rate": float(
                    partition.y.mean()
                ),
                "start_datetime": partition.metadata[
                    "TX_DATETIME"
                ].min(),
                "end_datetime": partition.metadata[
                    "TX_DATETIME"
                ].max(),
            }
        )

    return pd.DataFrame(rows)