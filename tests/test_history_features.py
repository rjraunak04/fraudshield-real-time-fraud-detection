"""Tests for FraudShield historical behavioural features."""

import pandas as pd
import pytest

from fraudshield.features.history import (
    HISTORICAL_FEATURE_COLUMNS,
    add_historical_behaviour_features,
)


def make_transactions() -> pd.DataFrame:
    """Create deterministic transactions for history tests."""
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2, 3, 4],
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 09:00:00",
                    "2018-04-01 10:00:00",
                    "2018-04-01 11:00:00",
                    "2018-04-01 12:00:00",
                ]
            ),
            "CUSTOMER_ID": [100, 100, 100, 200],
            "TERMINAL_ID": [10, 10, 20, 10],
            "TX_AMOUNT": [10.0, 20.0, 60.0, 40.0],
            "TX_FRAUD": [0, 0, 1, 0],
            "TX_FRAUD_SCENARIO": [0, 0, 2, 0],
        }
    )


def test_historical_features_are_created() -> None:
    """All configured historical features should be added."""
    result = add_historical_behaviour_features(
        make_transactions()
    )

    for column in HISTORICAL_FEATURE_COLUMNS:
        assert column in result.columns


def test_first_customer_transaction_has_zero_history() -> None:
    """First customer transaction should have no past baseline."""
    result = add_historical_behaviour_features(
        make_transactions()
    )

    first = result.loc[
        result["TRANSACTION_ID"] == 1
    ].iloc[0]

    assert first["CUSTOMER_HIST_TX_COUNT"] == 0
    assert first["CUSTOMER_HIST_MEAN_AMOUNT"] == pytest.approx(0.0)
    assert first["CUSTOMER_HIST_MAX_AMOUNT"] == pytest.approx(0.0)
    assert first["CUSTOMER_AMOUNT_TO_HIST_MEAN"] == pytest.approx(0.0)


def test_customer_history_uses_only_previous_transactions() -> None:
    """Customer statistics should exclude the current transaction."""
    result = add_historical_behaviour_features(
        make_transactions()
    )

    third = result.loc[
        result["TRANSACTION_ID"] == 3
    ].iloc[0]

    assert third["CUSTOMER_HIST_TX_COUNT"] == 2
    assert third["CUSTOMER_HIST_MEAN_AMOUNT"] == pytest.approx(15.0)
    assert third["CUSTOMER_HIST_MAX_AMOUNT"] == pytest.approx(20.0)
    assert third["CUSTOMER_AMOUNT_TO_HIST_MEAN"] == pytest.approx(4.0)


def test_terminal_history_is_independent_of_customer() -> None:
    """Terminal history should aggregate across prior customers."""
    result = add_historical_behaviour_features(
        make_transactions()
    )

    fourth = result.loc[
        result["TRANSACTION_ID"] == 4
    ].iloc[0]

    assert fourth["TERMINAL_HIST_TX_COUNT"] == 2
    assert fourth["TERMINAL_HIST_MEAN_AMOUNT"] == pytest.approx(15.0)

    assert fourth["TERMINAL_AMOUNT_TO_HIST_MEAN"] == pytest.approx(
        40.0 / 15.0
    )


def test_same_timestamp_customer_transactions_share_history() -> None:
    """Simultaneous customer transactions must not see one another."""
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
            "TERMINAL_ID": [10, 20, 30],
            "TX_AMOUNT": [10.0, 20.0, 30.0],
        }
    )

    result = add_historical_behaviour_features(
        transactions
    )

    second = result.loc[
        result["TRANSACTION_ID"] == 2
    ].iloc[0]

    third = result.loc[
        result["TRANSACTION_ID"] == 3
    ].iloc[0]

    assert second["CUSTOMER_HIST_TX_COUNT"] == 1
    assert third["CUSTOMER_HIST_TX_COUNT"] == 1

    assert second["CUSTOMER_HIST_MEAN_AMOUNT"] == pytest.approx(10.0)
    assert third["CUSTOMER_HIST_MEAN_AMOUNT"] == pytest.approx(10.0)


def test_same_timestamp_terminal_transactions_share_history() -> None:
    """Simultaneous terminal transactions must share prior state."""
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
            "CUSTOMER_ID": [100, 200, 300],
            "TERMINAL_ID": [10, 10, 10],
            "TX_AMOUNT": [10.0, 20.0, 30.0],
        }
    )

    result = add_historical_behaviour_features(
        transactions
    )

    second = result.loc[
        result["TRANSACTION_ID"] == 2
    ].iloc[0]

    third = result.loc[
        result["TRANSACTION_ID"] == 3
    ].iloc[0]

    assert second["TERMINAL_HIST_TX_COUNT"] == 1
    assert third["TERMINAL_HIST_TX_COUNT"] == 1

    assert second["TERMINAL_HIST_MEAN_AMOUNT"] == pytest.approx(10.0)
    assert third["TERMINAL_HIST_MEAN_AMOUNT"] == pytest.approx(10.0)


def test_different_entities_keep_independent_history() -> None:
    """New customers should not inherit another customer's history."""
    result = add_historical_behaviour_features(
        make_transactions()
    )

    fourth = result.loc[
        result["TRANSACTION_ID"] == 4
    ].iloc[0]

    assert fourth["CUSTOMER_HIST_TX_COUNT"] == 0
    assert fourth["CUSTOMER_HIST_MEAN_AMOUNT"] == pytest.approx(0.0)


def test_features_do_not_require_fraud_labels() -> None:
    """Historical features must not depend on target columns."""
    transactions = make_transactions().drop(
        columns=[
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
        ]
    )

    result = add_historical_behaviour_features(
        transactions
    )

    for column in HISTORICAL_FEATURE_COLUMNS:
        assert column in result.columns


def test_original_dataframe_is_not_modified() -> None:
    """Historical engineering should not mutate input data."""
    transactions = make_transactions()

    original = transactions.copy(deep=True)

    add_historical_behaviour_features(transactions)

    pd.testing.assert_frame_equal(
        transactions,
        original,
    )


def test_input_order_is_preserved() -> None:
    """Returned rows should preserve caller ordering."""
    transactions = make_transactions().iloc[
        [3, 1, 0, 2]
    ].reset_index(drop=True)

    expected_ids = transactions[
        "TRANSACTION_ID"
    ].tolist()

    result = add_historical_behaviour_features(
        transactions
    )

    assert result["TRANSACTION_ID"].tolist() == expected_ids


def test_duplicate_transaction_ids_are_rejected() -> None:
    """Transaction identifiers must remain unique."""
    transactions = make_transactions()

    transactions.loc[1, "TRANSACTION_ID"] = 1

    with pytest.raises(
        ValueError,
        match="TRANSACTION_ID must be unique",
    ):
        add_historical_behaviour_features(transactions)