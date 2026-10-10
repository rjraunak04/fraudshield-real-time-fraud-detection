"""Tests for FraudShield temporal exploratory analysis."""

import pandas as pd
import pytest

from fraudshield.analysis.temporal import (
    daily_fraud_summary,
    fraud_scenario_summary,
    hourly_fraud_summary,
    monthly_fraud_summary,
    target_summary,
    temporal_quality_summary,
)


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Create deterministic transactions for analysis tests."""
    return pd.DataFrame(
        {
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 10:00:00",
                    "2018-04-01 11:00:00",
                    "2018-04-02 10:30:00",
                    "2018-05-01 15:00:00",
                ]
            ),
            "TX_FRAUD": [0, 1, 0, 1],
            "TX_FRAUD_SCENARIO": [0, 1, 0, 2],
        }
    )


def test_target_summary_reports_class_distribution(
    transactions: pd.DataFrame,
) -> None:
    """Target summary should report correct class counts."""
    summary = target_summary(transactions)

    assert summary["transaction_count"].tolist() == [2, 2]
    assert summary["transaction_share"].tolist() == pytest.approx(
        [0.5, 0.5]
    )


def test_daily_summary_calculates_fraud_rate(
    transactions: pd.DataFrame,
) -> None:
    """Daily aggregation should calculate correct fraud rates."""
    summary = daily_fraud_summary(transactions)

    assert len(summary) == 3
    assert summary.loc[0, "transaction_count"] == 2
    assert summary.loc[0, "fraud_count"] == 1
    assert summary.loc[0, "fraud_rate"] == pytest.approx(0.5)


def test_monthly_summary_calculates_counts(
    transactions: pd.DataFrame,
) -> None:
    """Monthly aggregation should preserve transaction totals."""
    summary = monthly_fraud_summary(transactions)

    assert summary["transaction_count"].sum() == 4
    assert summary["fraud_count"].sum() == 2
    assert summary["month"].tolist() == [
        "2018-04",
        "2018-05",
    ]


def test_hourly_summary_preserves_transactions(
    transactions: pd.DataFrame,
) -> None:
    """Hourly aggregation should preserve all observations."""
    summary = hourly_fraud_summary(transactions)

    assert summary["transaction_count"].sum() == 4
    assert summary["fraud_count"].sum() == 2


def test_scenario_summary_preserves_transactions(
    transactions: pd.DataFrame,
) -> None:
    """Scenario diagnostics should preserve transaction totals."""
    summary = fraud_scenario_summary(transactions)

    assert summary["transaction_count"].sum() == 4

    scenario_two = summary.loc[
        summary["scenario"] == 2,
        "scenario_name",
    ].iloc[0]

    assert scenario_two == "compromised_terminal"


def test_temporal_quality_detects_sorted_data(
    transactions: pd.DataFrame,
) -> None:
    """Temporal quality should identify chronological ordering."""
    quality = temporal_quality_summary(transactions)

    assert quality["is_chronologically_sorted"] is True
    assert quality["n_calendar_days"] == 3
    assert quality["missing_calendar_gaps"] == 1


def test_temporal_quality_detects_unsorted_data(
    transactions: pd.DataFrame,
) -> None:
    """Temporal quality should detect nonchronological input."""
    unsorted = transactions.iloc[::-1].reset_index(drop=True)

    quality = temporal_quality_summary(unsorted)

    assert quality["is_chronologically_sorted"] is False


def test_analysis_rejects_empty_dataframe() -> None:
    """Temporal analysis should reject empty input."""
    with pytest.raises(
        ValueError,
        match="empty transaction dataset",
    ):
        target_summary(pd.DataFrame())