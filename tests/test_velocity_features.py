"""Tests for FraudShield customer velocity features."""

import pandas as pd
import pytest

from fraudshield.features.velocity import (
    VELOCITY_FEATURE_COLUMNS,
    add_customer_velocity_features,
)


def make_transactions() -> pd.DataFrame:
    """Create deterministic customer transactions."""
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2, 3, 4, 5],
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 10:00:00",
                    "2018-04-01 10:30:00",
                    "2018-04-01 12:00:00",
                    "2018-04-02 09:00:00",
                    "2018-04-01 10:15:00",
                ]
            ),
            "CUSTOMER_ID": [
                100,
                100,
                100,
                100,
                200,
            ],
            "TX_AMOUNT": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ],
            "TX_FRAUD": [0, 0, 1, 0, 1],
            "TX_FRAUD_SCENARIO": [0, 0, 2, 0, 3],
        }
    )


def test_velocity_features_are_created() -> None:
    """All configured velocity features should be added."""
    result = add_customer_velocity_features(
        make_transactions()
    )

    for column in VELOCITY_FEATURE_COLUMNS:
        assert column in result.columns


def test_first_customer_transaction_has_zero_history() -> None:
    """A customer's first transaction should have no history."""
    result = add_customer_velocity_features(
        make_transactions()
    )

    first = result.loc[
        result["TRANSACTION_ID"] == 1
    ].iloc[0]

    assert first["CUSTOMER_TX_COUNT_1H"] == 0
    assert first["CUSTOMER_TX_AMOUNT_1H"] == pytest.approx(0.0)
    assert first["CUSTOMER_TX_COUNT_24H"] == 0
    assert first["CUSTOMER_TX_AMOUNT_24H"] == pytest.approx(0.0)


def test_one_hour_velocity_uses_only_recent_past() -> None:
    """One-hour window should use only preceding transactions."""
    result = add_customer_velocity_features(
        make_transactions()
    )

    second = result.loc[
        result["TRANSACTION_ID"] == 2
    ].iloc[0]

    assert second["CUSTOMER_TX_COUNT_1H"] == 1
    assert second["CUSTOMER_TX_AMOUNT_1H"] == pytest.approx(10.0)

    third = result.loc[
        result["TRANSACTION_ID"] == 3
    ].iloc[0]

    assert third["CUSTOMER_TX_COUNT_1H"] == 0
    assert third["CUSTOMER_TX_AMOUNT_1H"] == pytest.approx(0.0)


def test_24_hour_velocity_uses_customer_history() -> None:
    """Twenty-four-hour features should retain recent history."""
    result = add_customer_velocity_features(
        make_transactions()
    )

    third = result.loc[
        result["TRANSACTION_ID"] == 3
    ].iloc[0]

    assert third["CUSTOMER_TX_COUNT_24H"] == 2
    assert third["CUSTOMER_TX_AMOUNT_24H"] == pytest.approx(30.0)


def test_customers_have_independent_history() -> None:
    """One customer's transactions must not affect another."""
    result = add_customer_velocity_features(
        make_transactions()
    )

    customer_two = result.loc[
        result["TRANSACTION_ID"] == 5
    ].iloc[0]

    assert customer_two["CUSTOMER_TX_COUNT_1H"] == 0
    assert customer_two["CUSTOMER_TX_AMOUNT_24H"] == pytest.approx(0.0)


def test_same_timestamp_transactions_do_not_see_each_other() -> None:
    """Transactions at the same timestamp should share prior history."""
    transactions = pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2, 3],
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 09:00:00",
                    "2018-04-01 10:00:00",
                    "2018-04-01 10:00:00",
                ]
            ),
            "CUSTOMER_ID": [100, 100, 100],
            "TX_AMOUNT": [10.0, 20.0, 30.0],
        }
    )

    result = add_customer_velocity_features(
        transactions
    )

    second = result.loc[
        result["TRANSACTION_ID"] == 2
    ].iloc[0]

    third = result.loc[
        result["TRANSACTION_ID"] == 3
    ].iloc[0]

    assert second["CUSTOMER_TX_COUNT_1H"] == 1
    assert third["CUSTOMER_TX_COUNT_1H"] == 1

    assert second["CUSTOMER_TX_AMOUNT_1H"] == pytest.approx(10.0)
    assert third["CUSTOMER_TX_AMOUNT_1H"] == pytest.approx(10.0)


def test_velocity_does_not_require_target_columns() -> None:
    """Velocity features must not depend on fraud labels."""
    transactions = make_transactions().drop(
        columns=[
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
        ]
    )

    result = add_customer_velocity_features(
        transactions
    )

    for column in VELOCITY_FEATURE_COLUMNS:
        assert column in result.columns


def test_original_dataframe_is_not_modified() -> None:
    """Velocity engineering should not mutate input data."""
    transactions = make_transactions()

    original = transactions.copy(deep=True)

    add_customer_velocity_features(transactions)

    pd.testing.assert_frame_equal(
        transactions,
        original,
    )


def test_velocity_preserves_input_row_order() -> None:
    """Returned transactions should preserve caller row ordering."""
    transactions = make_transactions().iloc[
        [4, 2, 0, 3, 1]
    ].reset_index(drop=True)

    original_ids = transactions[
        "TRANSACTION_ID"
    ].tolist()

    result = add_customer_velocity_features(
        transactions
    )

    assert result["TRANSACTION_ID"].tolist() == original_ids


def test_velocity_rejects_missing_customer_id() -> None:
    """CUSTOMER_ID is required for customer history."""
    transactions = make_transactions().drop(
        columns=["CUSTOMER_ID"]
    )

    with pytest.raises(
        ValueError,
        match="CUSTOMER_ID",
    ):
        add_customer_velocity_features(transactions)


def test_velocity_rejects_duplicate_transaction_ids() -> None:
    """Velocity computation requires unique transaction IDs."""
    transactions = make_transactions()

    transactions.loc[1, "TRANSACTION_ID"] = 1

    with pytest.raises(
        ValueError,
        match="TRANSACTION_ID must be unique",
    ):
        add_customer_velocity_features(transactions)