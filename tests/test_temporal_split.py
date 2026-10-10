"""Tests for FraudShield leakage-safe temporal splitting."""

import pandas as pd
import pytest

from fraudshield.data.split import (
    TemporalSplit,
    split_summary,
    temporal_train_validation_test_split,
    validate_temporal_split,
)


def make_transactions(
    n_rows: int = 100,
) -> pd.DataFrame:
    """Create deterministic chronological transactions."""
    timestamps = pd.date_range(
        "2018-01-01",
        periods=n_rows,
        freq="h",
    )

    return pd.DataFrame(
        {
            "TRANSACTION_ID": range(n_rows),
            "TX_DATETIME": timestamps,
            "TX_FRAUD": [
                index % 10 == 0
                for index in range(n_rows)
            ],
        }
    ).assign(
        TX_FRAUD=lambda frame: frame[
            "TX_FRAUD"
        ].astype(int)
    )


def test_temporal_split_has_expected_sizes() -> None:
    """Split should follow configured chronological fractions."""
    transactions = make_transactions(100)

    split = temporal_train_validation_test_split(
        transactions
    )

    assert len(split.train) == 70
    assert len(split.validation) == 15
    assert len(split.test) == 15


def test_temporal_split_preserves_all_rows() -> None:
    """No transaction should disappear during splitting."""
    transactions = make_transactions(100)

    split = temporal_train_validation_test_split(
        transactions
    )

    total = (
        len(split.train)
        + len(split.validation)
        + len(split.test)
    )

    assert total == len(transactions)


def test_temporal_split_orders_unsorted_input() -> None:
    """Input ordering should not affect chronological partitions."""
    transactions = (
        make_transactions(100)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    split = temporal_train_validation_test_split(
        transactions
    )

    assert (
        split.train["TX_DATETIME"].max()
        <= split.validation["TX_DATETIME"].min()
    )

    assert (
        split.validation["TX_DATETIME"].max()
        <= split.test["TX_DATETIME"].min()
    )


def test_temporal_split_has_no_id_overlap() -> None:
    """Transaction IDs must remain disjoint across partitions."""
    split = temporal_train_validation_test_split(
        make_transactions(100)
    )

    train_ids = set(split.train["TRANSACTION_ID"])
    validation_ids = set(
        split.validation["TRANSACTION_ID"]
    )
    test_ids = set(split.test["TRANSACTION_ID"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_temporal_split_validation_passes() -> None:
    """A correctly constructed temporal split should validate."""
    split = temporal_train_validation_test_split(
        make_transactions(100)
    )

    validate_temporal_split(split)


def test_temporal_split_rejects_invalid_fractions() -> None:
    """Invalid split fractions should fail clearly."""
    transactions = make_transactions(100)

    with pytest.raises(
        ValueError,
        match="less than 1",
    ):
        temporal_train_validation_test_split(
            transactions,
            train_fraction=0.80,
            validation_fraction=0.20,
        )


def test_temporal_split_rejects_empty_input() -> None:
    """Empty transaction data should not be split."""
    empty = pd.DataFrame(
        columns=[
            "TRANSACTION_ID",
            "TX_DATETIME",
            "TX_FRAUD",
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty transaction dataset",
    ):
        temporal_train_validation_test_split(empty)


def test_validation_detects_partition_overlap() -> None:
    """Validator should detect repeated IDs across partitions."""
    transactions = make_transactions(100)

    split = temporal_train_validation_test_split(
        transactions
    )

    broken_validation = pd.concat(
        [
            split.validation,
            split.train.iloc[[0]],
        ],
        ignore_index=True,
    )

    broken = TemporalSplit(
        train=split.train,
        validation=broken_validation,
        test=split.test,
    )

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        validate_temporal_split(broken)


def test_split_summary_preserves_row_count() -> None:
    """Summary row counts should equal the original dataset size."""
    split = temporal_train_validation_test_split(
        make_transactions(100)
    )

    summary = split_summary(split)

    assert summary["rows"].sum() == 100
    assert summary["partition"].tolist() == [
        "train",
        "validation",
        "test",
    ]