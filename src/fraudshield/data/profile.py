"""Dataset profiling utilities for FraudShield."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_dataset_profile(
    transactions: pd.DataFrame,
) -> dict[str, Any]:
    """Build a compact analytical profile of transaction data."""
    if transactions.empty:
        raise ValueError("Cannot profile an empty transaction dataset.")

    fraud_count = int(transactions["TX_FRAUD"].sum())
    transaction_count = len(transactions)

    return {
        "n_transactions": transaction_count,
        "n_columns": transactions.shape[1],
        "n_customers": int(
            transactions["CUSTOMER_ID"].nunique()
        ),
        "n_terminals": int(
            transactions["TERMINAL_ID"].nunique()
        ),
        "start_datetime": transactions["TX_DATETIME"].min(),
        "end_datetime": transactions["TX_DATETIME"].max(),
        "fraud_count": fraud_count,
        "legitimate_count": transaction_count - fraud_count,
        "fraud_rate": fraud_count / transaction_count,
        "missing_values": int(
            transactions.isna().sum().sum()
        ),
        "duplicate_transaction_ids": int(
            transactions["TRANSACTION_ID"].duplicated().sum()
        ),
        "amount_min": float(
            transactions["TX_AMOUNT"].min()
        ),
        "amount_median": float(
            transactions["TX_AMOUNT"].median()
        ),
        "amount_mean": float(
            transactions["TX_AMOUNT"].mean()
        ),
        "amount_max": float(
            transactions["TX_AMOUNT"].max()
        ),
    }


def profile_to_frame(
    profile: dict[str, Any],
) -> pd.DataFrame:
    """Convert a dataset profile dictionary into a readable table."""
    return pd.DataFrame(
        {
            "metric": list(profile.keys()),
            "value": list(profile.values()),
        }
    )