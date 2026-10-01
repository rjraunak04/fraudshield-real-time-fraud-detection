"""Data-contract validation for FraudShield transaction datasets."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
}


def validate_transactions(df: pd.DataFrame) -> None:
    """Validate the FraudShield transaction data contract.

    Raises
    ------
    ValueError
        If the dataset violates an expected contract.
    """
    if df.empty:
        raise ValueError("Transaction dataset must not be empty.")

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    if df["TRANSACTION_ID"].isna().any():
        raise ValueError("TRANSACTION_ID must not contain missing values.")

    if df["TRANSACTION_ID"].duplicated().any():
        raise ValueError("TRANSACTION_ID must be unique.")

    if df["CUSTOMER_ID"].isna().any():
        raise ValueError("CUSTOMER_ID must not contain missing values.")

    if df["TERMINAL_ID"].isna().any():
        raise ValueError("TERMINAL_ID must not contain missing values.")

    if df["TX_AMOUNT"].isna().any():
        raise ValueError("TX_AMOUNT must not contain missing values.")

    if (df["TX_AMOUNT"] < 0).any():
        raise ValueError("TX_AMOUNT must not contain negative values.")

    valid_fraud_labels = {0, 1}

    observed_fraud_labels = set(
        df["TX_FRAUD"].dropna().unique()
    )

    if not observed_fraud_labels.issubset(valid_fraud_labels):
        raise ValueError("TX_FRAUD must contain only binary values 0 and 1.")

    if df["TX_FRAUD"].isna().any():
        raise ValueError("TX_FRAUD must not contain missing values.")

    if df["TX_FRAUD"].nunique() < 2:
        raise ValueError(
            "TX_FRAUD must contain both legitimate and fraudulent classes."
        )

    parsed_datetime = pd.to_datetime(
        df["TX_DATETIME"],
        errors="coerce",
        format="mixed",
    )

    if parsed_datetime.isna().any():
        raise ValueError("TX_DATETIME contains invalid datetime values.")

    if (df["TX_TIME_SECONDS"] < 0).any():
        raise ValueError("TX_TIME_SECONDS must not contain negative values.")

    if (df["TX_TIME_DAYS"] < 0).any():
        raise ValueError("TX_TIME_DAYS must not contain negative values.")