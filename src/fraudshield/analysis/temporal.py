"""Temporal exploratory analysis utilities for FraudShield."""

from __future__ import annotations

import pandas as pd

REQUIRED_ANALYSIS_COLUMNS = {
    "TX_DATETIME",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
}


def _validate_analysis_input(transactions: pd.DataFrame) -> None:
    """Validate columns required for temporal fraud analysis."""
    if transactions.empty:
        raise ValueError("Cannot analyse an empty transaction dataset.")

    missing = REQUIRED_ANALYSIS_COLUMNS.difference(
        transactions.columns
    )

    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing columns required for temporal analysis: {columns}"
        )


def target_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise legitimate and fraudulent transaction classes."""
    _validate_analysis_input(transactions)

    counts = (
        transactions["TX_FRAUD"]
        .value_counts()
        .sort_index()
    )

    total = len(transactions)

    summary = pd.DataFrame(
        {
            "TX_FRAUD": counts.index,
            "transaction_count": counts.values,
        }
    )

    summary["transaction_share"] = (
        summary["transaction_count"] / total
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
            "transaction_share",
        ]
    ]


def daily_fraud_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate transaction volume and fraud behaviour by calendar day."""
    _validate_analysis_input(transactions)

    frame = transactions[
        ["TX_DATETIME", "TX_FRAUD"]
    ].copy()

    frame["date"] = frame["TX_DATETIME"].dt.normalize()

    daily = (
        frame.groupby("date", as_index=False)
        .agg(
            transaction_count=("TX_FRAUD", "size"),
            fraud_count=("TX_FRAUD", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["legitimate_count"] = (
        daily["transaction_count"] - daily["fraud_count"]
    )

    daily["fraud_rate"] = (
        daily["fraud_count"] / daily["transaction_count"]
    )

    return daily


def monthly_fraud_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate transaction volume and fraud behaviour by month."""
    _validate_analysis_input(transactions)

    frame = transactions[
        ["TX_DATETIME", "TX_FRAUD"]
    ].copy()

    frame["month"] = (
        frame["TX_DATETIME"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        frame.groupby("month", as_index=False)
        .agg(
            transaction_count=("TX_FRAUD", "size"),
            fraud_count=("TX_FRAUD", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    monthly["fraud_rate"] = (
        monthly["fraud_count"]
        / monthly["transaction_count"]
    )

    return monthly


def hourly_fraud_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate transaction volume and fraud behaviour by hour of day."""
    _validate_analysis_input(transactions)

    frame = transactions[
        ["TX_DATETIME", "TX_FRAUD"]
    ].copy()

    frame["hour"] = frame["TX_DATETIME"].dt.hour

    hourly = (
        frame.groupby("hour", as_index=False)
        .agg(
            transaction_count=("TX_FRAUD", "size"),
            fraud_count=("TX_FRAUD", "sum"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )

    hourly["fraud_rate"] = (
        hourly["fraud_count"]
        / hourly["transaction_count"]
    )

    return hourly


def fraud_scenario_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise simulator fraud scenarios for diagnostics only."""
    _validate_analysis_input(transactions)

    scenario_names = {
        0: "legitimate",
        1: "high_amount",
        2: "compromised_terminal",
        3: "compromised_customer",
    }

    counts = (
        transactions["TX_FRAUD_SCENARIO"]
        .value_counts()
        .sort_index()
    )

    summary = pd.DataFrame(
        {
            "scenario": counts.index,
            "transaction_count": counts.values,
        }
    )

    summary["scenario_name"] = summary["scenario"].map(
        scenario_names
    )

    summary["transaction_share"] = (
        summary["transaction_count"] / len(transactions)
    )

    return summary[
        [
            "scenario",
            "scenario_name",
            "transaction_count",
            "transaction_share",
        ]
    ]


def temporal_quality_summary(
    transactions: pd.DataFrame,
) -> dict[str, object]:
    """Report basic temporal quality properties of the dataset."""
    _validate_analysis_input(transactions)

    timestamps = transactions["TX_DATETIME"]

    dates = timestamps.dt.normalize()

    unique_dates = pd.Index(dates.unique()).sort_values()

    if len(unique_dates) > 1:
        date_differences = unique_dates.to_series().diff().dropna()

        missing_calendar_gaps = int(
            (date_differences > pd.Timedelta(days=1)).sum()
        )
    else:
        missing_calendar_gaps = 0

    return {
        "start_datetime": timestamps.min(),
        "end_datetime": timestamps.max(),
        "n_calendar_days": int(dates.nunique()),
        "is_chronologically_sorted": bool(
            timestamps.is_monotonic_increasing
        ),
        "missing_calendar_gaps": missing_calendar_gaps,
    }