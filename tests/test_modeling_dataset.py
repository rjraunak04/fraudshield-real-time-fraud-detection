"""Tests for FraudShield modeling dataset construction."""

import pandas as pd
import pytest

from fraudshield.modeling.dataset import (
    FORBIDDEN_MODEL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ModelingDataset,
    ModelingPartition,
    build_modeling_dataset,
    modeling_dataset_summary,
    validate_modeling_dataset,
)


def make_transactions(
    n_rows: int = 40,
) -> pd.DataFrame:
    """Create deterministic transactions with both classes per split."""
    timestamps = pd.date_range(
        "2018-04-01",
        periods=n_rows,
        freq="h",
    )

    fraud = [
        1 if index % 5 == 0 else 0
        for index in range(n_rows)
    ]

    return pd.DataFrame(
        {
            "TRANSACTION_ID": range(n_rows),
            "TX_DATETIME": timestamps,
            "CUSTOMER_ID": [
                index % 5
                for index in range(n_rows)
            ],
            "TERMINAL_ID": [
                index % 7
                for index in range(n_rows)
            ],
            "TX_AMOUNT": [
                float(10 + index)
                for index in range(n_rows)
            ],
            "TX_TIME_SECONDS": [
                index * 3600
                for index in range(n_rows)
            ],
            "TX_TIME_DAYS": [
                index // 24
                for index in range(n_rows)
            ],
            "TX_FRAUD": fraud,
            "TX_FRAUD_SCENARIO": [
                2 if value else 0
                for value in fraud
            ],
        }
    )


def test_modeling_dataset_uses_chronological_split() -> None:
    """Train, validation, and test should remain chronological."""
    dataset = build_modeling_dataset(
        make_transactions()
    )

    assert (
        dataset.train.metadata[
            "TX_DATETIME"
        ].max()
        <= dataset.validation.metadata[
            "TX_DATETIME"
        ].min()
    )

    assert (
        dataset.validation.metadata[
            "TX_DATETIME"
        ].max()
        <= dataset.test.metadata[
            "TX_DATETIME"
        ].min()
    )


def test_modeling_dataset_has_expected_partition_sizes() -> None:
    """Default temporal split should use 70/15/15 proportions."""
    dataset = build_modeling_dataset(
        make_transactions(40)
    )

    assert len(dataset.train.X) == 28
    assert len(dataset.validation.X) == 6
    assert len(dataset.test.X) == 6


def test_modeling_dataset_uses_explicit_feature_contract() -> None:
    """Every partition should use the approved feature allowlist."""
    dataset = build_modeling_dataset(
        make_transactions()
    )

    assert (
        dataset.train.X.columns.tolist()
        == MODEL_FEATURE_COLUMNS
    )

    assert (
        dataset.validation.X.columns.tolist()
        == MODEL_FEATURE_COLUMNS
    )

    assert (
        dataset.test.X.columns.tolist()
        == MODEL_FEATURE_COLUMNS
    )


def test_forbidden_columns_are_excluded() -> None:
    """Identifiers, target, and simulator fields must not enter X."""
    dataset = build_modeling_dataset(
        make_transactions()
    )

    for column in FORBIDDEN_MODEL_COLUMNS:
        assert column not in dataset.train.X.columns


def test_target_is_separated_from_features() -> None:
    """Fraud target should be stored independently from X."""
    dataset = build_modeling_dataset(
        make_transactions()
    )

    assert TARGET_COLUMN not in dataset.train.X.columns
    assert set(dataset.train.y.unique()) == {0, 1}


def test_metadata_preserves_transaction_identity() -> None:
    """Transaction identifiers should remain available as metadata."""
    dataset = build_modeling_dataset(
        make_transactions()
    )

    assert "TRANSACTION_ID" in dataset.train.metadata.columns
    assert "TX_DATETIME" in dataset.train.metadata.columns


def test_modeling_partitions_do_not_overlap() -> None:
    """Transactions should belong to exactly one partition."""
    dataset = build_modeling_dataset(
        make_transactions()
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

    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_modeling_dataset_has_no_missing_features() -> None:
    """Approved model features should be fully populated."""
    dataset = build_modeling_dataset(
        make_transactions()
    )

    assert not dataset.train.X.isna().any().any()
    assert not dataset.validation.X.isna().any().any()
    assert not dataset.test.X.isna().any().any()


def test_modeling_dataset_rejects_non_binary_target() -> None:
    """Fraud target should contain only zero and one."""
    transactions = make_transactions()

    transactions.loc[0, "TX_FRAUD"] = 2

    with pytest.raises(
        ValueError,
        match="binary values",
    ):
        build_modeling_dataset(transactions)


def test_modeling_dataset_rejects_missing_target() -> None:
    """Supervised modeling requires TX_FRAUD."""
    transactions = make_transactions().drop(
        columns=["TX_FRAUD"]
    )

    with pytest.raises(
        ValueError,
        match="Missing columns required for modeling",
    ):
        build_modeling_dataset(transactions)


def test_modeling_validation_detects_forbidden_feature() -> None:
    """Validation should reject target leakage in feature matrices."""
    dataset = build_modeling_dataset(
        make_transactions()
    )

    bad_train_x = dataset.train.X.copy()

    bad_train_x["TX_FRAUD"] = (
        dataset.train.y.to_numpy()
    )

    bad_dataset = ModelingDataset(
        train=ModelingPartition(
            X=bad_train_x,
            y=dataset.train.y,
            metadata=dataset.train.metadata,
        ),
        validation=dataset.validation,
        test=dataset.test,
    )

    with pytest.raises(
        ValueError,
        match="model feature contract",
    ):
        validate_modeling_dataset(
            bad_dataset
        )


def test_modeling_summary_has_three_partitions() -> None:
    """Summary should contain train, validation, and test."""
    dataset = build_modeling_dataset(
        make_transactions()
    )

    summary = modeling_dataset_summary(
        dataset
    )

    assert summary["partition"].tolist() == [
        "train",
        "validation",
        "test",
    ]