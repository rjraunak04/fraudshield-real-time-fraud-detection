"""Behavioural exploratory analysis utilities for FraudShield."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
}


def _validate_input(transactions: pd.DataFrame) -> None:
    """Validate columns required for behavioural analysis."""
    if transactions.empty:
        raise ValueError("Cannot analyse an empty transaction dataset.")

    missing = REQUIRED_COLUMNS.difference(transactions.columns)

    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing columns required for behavioural analysis: {columns}"
        )


def amount_by_class(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise transaction amounts for each fraud class."""
    _validate_input(transactions)

    summary = (
        transactions.groupby("TX_FRAUD")["TX_AMOUNT"]
        .agg(
            transaction_count="size",
            mean_amount="mean",
            median_amount="median",
            std_amount="std",
            min_amount="min",
            max_amount="max",
        )
        .reset_index()
    )

    summary["class_name"] = summary["TX_FRAUD"].map(
        {
            0: "legitimate",
            1: "fraud",
        }
    )

    return summary[
        [
            "TX_FRAUD",
            "class_name",
            "transaction_count",
            "mean_amount",
            "median_amount",
            "std_amount",
            "min_amount",
            "max_amount",
        ]
    ]


def amount_quantile_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare amount quantiles between legitimate and fraud classes."""
    _validate_input(transactions)

    quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    rows: list[dict[str, float | int]] = []

    for fraud_label, group in transactions.groupby("TX_FRAUD"):
        values = group["TX_AMOUNT"].quantile(quantiles)

        for quantile, amount in values.items():
            rows.append(
                {
                    "TX_FRAUD": int(fraud_label),
                    "quantile": float(quantile),
                    "amount": float(amount),
                }
            )

    return pd.DataFrame(rows)


def amount_band_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate fraud behaviour across transaction amount bands."""
    _validate_input(transactions)

    frame = transactions[
        ["TX_AMOUNT", "TX_FRAUD"]
    ].copy()

    bins = [-np.inf, 25, 50, 100, 150, 200, np.inf]

    labels = [
        "0-25",
        "25-50",
        "50-100",
        "100-150",
        "150-200",
        "200+",
    ]

    frame["amount_band"] = pd.cut(
        frame["TX_AMOUNT"],
        bins=bins,
        labels=labels,
        right=False,
    )

    summary = (
        frame.groupby(
            "amount_band",
            observed=True,
        )
        .agg(
            transaction_count=("TX_FRAUD", "size"),
            fraud_count=("TX_FRAUD", "sum"),
        )
        .reset_index()
    )

    summary["fraud_rate"] = (
        summary["fraud_count"]
        / summary["transaction_count"]
    )

    return summary


def customer_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Build customer-level descriptive behavioural statistics."""
    _validate_input(transactions)

    summary = (
        transactions.groupby("CUSTOMER_ID")
        .agg(
            transaction_count=("TX_FRAUD", "size"),
            fraud_count=("TX_FRAUD", "sum"),
            total_amount=("TX_AMOUNT", "sum"),
            mean_amount=("TX_AMOUNT", "mean"),
            median_amount=("TX_AMOUNT", "median"),
            unique_terminals=("TERMINAL_ID", "nunique"),
        )
        .reset_index()
    )

    summary["fraud_rate"] = (
        summary["fraud_count"]
        / summary["transaction_count"]
    )

    return summary


def terminal_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Build terminal-level descriptive behavioural statistics."""
    _validate_input(transactions)

    summary = (
        transactions.groupby("TERMINAL_ID")
        .agg(
            transaction_count=("TX_FRAUD", "size"),
            fraud_count=("TX_FRAUD", "sum"),
            total_amount=("TX_AMOUNT", "sum"),
            mean_amount=("TX_AMOUNT", "mean"),
            unique_customers=("CUSTOMER_ID", "nunique"),
        )
        .reset_index()
    )

    summary["fraud_rate"] = (
        summary["fraud_count"]
        / summary["transaction_count"]
    )

    return summary


def scenario_amount_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Describe transaction amounts by simulator scenario."""
    _validate_input(transactions)

    summary = (
        transactions.groupby("TX_FRAUD_SCENARIO")
        .agg(
            transaction_count=("TX_AMOUNT", "size"),
            mean_amount=("TX_AMOUNT", "mean"),
            median_amount=("TX_AMOUNT", "median"),
            max_amount=("TX_AMOUNT", "max"),
        )
        .reset_index()
    )

    return summary