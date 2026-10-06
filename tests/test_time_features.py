"""Tests for FraudShield transaction-time features."""

import pandas as pd
import pytest

from fraudshield.features.time import (
    TIME_FEATURE_COLUMNS,
    add_transaction_time_features,
)


def make_transactions() -> pd.DataFrame:
    """Create deterministic timestamps for feature tests."""
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2, 3],
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-02 02:30:00",
                    "2018-04-07 12:15:00",
                    "2018-05-06 20:45:00",
                ]
            ),
            "TX_FRAUD": [0, 1, 0],
            "TX_FRAUD_SCENARIO": [0, 2, 0],
        }
    )


def test_time_features_are_created() -> None:
    """All configured time features should be added."""
    result = add_transaction_time_features(
        make_transactions()
    )

    for column in TIME_FEATURE_COLUMNS:
        assert column in result.columns


def test_hour_feature_is_correct() -> None:
    """Transaction hour should come from TX_DATETIME."""
    result = add_transaction_time_features(
        make_transactions()
    )

    assert result["TX_HOUR"].tolist() == [
        2,
        12,
        20,
    ]


def test_day_of_week_is_correct() -> None:
    """Monday should be zero and Sunday six."""
    result = add_transaction_time_features(
        make_transactions()
    )

    assert result["TX_DAY_OF_WEEK"].tolist() == [
        0,
        5,
        6,
    ]


def test_weekend_feature_is_correct() -> None:
    """Saturday and Sunday should be weekend transactions."""
    result = add_transaction_time_features(
        make_transactions()
    )

    assert result["TX_IS_WEEKEND"].tolist() == [
        0,
        1,
        1,
    ]


def test_night_feature_is_correct() -> None:
    """Hours from midnight through 05:59 should be night."""
    result = add_transaction_time_features(
        make_transactions()
    )

    assert result["TX_IS_NIGHT"].tolist() == [
        1,
        0,
        0,
    ]


def test_business_hours_feature_is_correct() -> None:
    """Hours 09 through 16 should be business hours."""
    result = add_transaction_time_features(
        make_transactions()
    )

    assert result[
        "TX_IS_BUSINESS_HOURS"
    ].tolist() == [
        0,
        1,
        0,
    ]


def test_original_dataframe_is_not_modified() -> None:
    """Feature engineering should not mutate input data."""
    transactions = make_transactions()

    original_columns = transactions.columns.tolist()

    add_transaction_time_features(transactions)

    assert transactions.columns.tolist() == original_columns


def test_time_features_do_not_use_target_columns() -> None:
    """Features should work without fraud labels."""
    transactions = make_transactions().drop(
        columns=[
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
        ]
    )

    result = add_transaction_time_features(
        transactions
    )

    for column in TIME_FEATURE_COLUMNS:
        assert column in result.columns


def test_time_features_reject_missing_datetime() -> None:
    """Feature engineering should require TX_DATETIME."""
    transactions = make_transactions().drop(
        columns=["TX_DATETIME"]
    )

    with pytest.raises(
        ValueError,
        match="TX_DATETIME is required",
    ):
        add_transaction_time_features(transactions)


def test_time_features_reject_empty_dataset() -> None:
    """Feature engineering should reject empty input."""
    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        add_transaction_time_features(
            pd.DataFrame()
        )