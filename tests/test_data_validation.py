"""Tests for the FraudShield transaction data contract."""

import pandas as pd
import pytest

from fraudshield.data.validate import validate_transactions


def make_valid_dataframe() -> pd.DataFrame:
    """Create a minimal valid FraudShield transaction dataset."""
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [0, 1],
            "TX_DATETIME": [
                "2018-04-01 10:00:00",
                "2018-04-01 11:00:00",
            ],
            "CUSTOMER_ID": [100, 101],
            "TERMINAL_ID": [10, 11],
            "TX_AMOUNT": [25.50, 250.00],
            "TX_TIME_SECONDS": [36000, 39600],
            "TX_TIME_DAYS": [0, 0],
            "TX_FRAUD": [0, 1],
            "TX_FRAUD_SCENARIO": [0, 1],
        }
    )


def test_valid_dataframe_passes() -> None:
    """A valid transaction dataset should pass validation."""
    df = make_valid_dataframe()

    validate_transactions(df)


def test_missing_required_column_fails() -> None:
    """Missing required columns should violate the data contract."""
    df = make_valid_dataframe().drop(columns=["TX_AMOUNT"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_transactions(df)


def test_invalid_fraud_label_fails() -> None:
    """Fraud labels must remain binary."""
    df = make_valid_dataframe()
    df.loc[1, "TX_FRAUD"] = 2

    with pytest.raises(ValueError, match="binary values"):
        validate_transactions(df)


def test_negative_amount_fails() -> None:
    """Negative transaction amounts should be rejected."""
    df = make_valid_dataframe()
    df.loc[0, "TX_AMOUNT"] = -10.0

    with pytest.raises(ValueError, match="negative"):
        validate_transactions(df)


def test_duplicate_transaction_id_fails() -> None:
    """Transaction identifiers must be unique."""
    df = make_valid_dataframe()
    df.loc[1, "TRANSACTION_ID"] = 0

    with pytest.raises(ValueError, match="unique"):
        validate_transactions(df)


def test_invalid_datetime_fails() -> None:
    """Invalid transaction timestamps should be rejected."""
    df = make_valid_dataframe()
    df.loc[0, "TX_DATETIME"] = "not-a-date"

    with pytest.raises(ValueError, match="invalid datetime"):
        validate_transactions(df)


def test_single_target_class_fails() -> None:
    """Both legitimate and fraudulent classes must be present."""
    df = make_valid_dataframe()
    df["TX_FRAUD"] = 0

    with pytest.raises(ValueError, match="both legitimate and fraudulent"):
        validate_transactions(df)