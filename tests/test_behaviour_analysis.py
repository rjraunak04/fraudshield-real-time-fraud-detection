"""Tests for FraudShield behavioural exploratory analysis."""

import pandas as pd
import pytest

from fraudshield.analysis.behaviour import (
    amount_band_summary,
    amount_by_class,
    amount_quantile_summary,
    customer_summary,
    scenario_amount_summary,
    terminal_summary,
)


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Create deterministic transactions for behavioural tests."""
    return pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 1, 2, 2, 3],
            "TERMINAL_ID": [10, 11, 10, 12, 12],
            "TX_AMOUNT": [10.0, 250.0, 40.0, 120.0, 210.0],
            "TX_FRAUD": [0, 1, 0, 0, 1],
            "TX_FRAUD_SCENARIO": [0, 1, 0, 0, 2],
        }
    )


def test_amount_by_class_preserves_transactions(
    transactions: pd.DataFrame,
) -> None:
    """Amount class summary should preserve transaction count."""
    summary = amount_by_class(transactions)

    assert summary["transaction_count"].sum() == 5
    assert set(summary["class_name"]) == {
        "legitimate",
        "fraud",
    }


def test_amount_quantiles_include_expected_levels(
    transactions: pd.DataFrame,
) -> None:
    """Amount quantiles should be produced for both classes."""
    summary = amount_quantile_summary(transactions)

    assert set(summary["TX_FRAUD"]) == {0, 1}
    assert set(summary["quantile"]) == {
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    }


def test_amount_bands_preserve_transactions(
    transactions: pd.DataFrame,
) -> None:
    """Amount bands should preserve all transactions."""
    summary = amount_band_summary(transactions)

    assert summary["transaction_count"].sum() == 5
    assert summary["fraud_count"].sum() == 2


def test_customer_summary_preserves_fraud_count(
    transactions: pd.DataFrame,
) -> None:
    """Customer aggregation should preserve fraud totals."""
    summary = customer_summary(transactions)

    assert len(summary) == 3
    assert summary["transaction_count"].sum() == 5
    assert summary["fraud_count"].sum() == 2


def test_customer_summary_counts_unique_terminals(
    transactions: pd.DataFrame,
) -> None:
    """Customer statistics should count distinct terminals."""
    summary = customer_summary(transactions)

    customer_one = summary.loc[
        summary["CUSTOMER_ID"] == 1
    ].iloc[0]

    assert customer_one["unique_terminals"] == 2


def test_terminal_summary_preserves_fraud_count(
    transactions: pd.DataFrame,
) -> None:
    """Terminal aggregation should preserve fraud totals."""
    summary = terminal_summary(transactions)

    assert len(summary) == 3
    assert summary["transaction_count"].sum() == 5
    assert summary["fraud_count"].sum() == 2


def test_terminal_summary_counts_unique_customers(
    transactions: pd.DataFrame,
) -> None:
    """Terminal statistics should count distinct customers."""
    summary = terminal_summary(transactions)

    terminal_ten = summary.loc[
        summary["TERMINAL_ID"] == 10
    ].iloc[0]

    assert terminal_ten["unique_customers"] == 2


def test_scenario_amount_summary_preserves_transactions(
    transactions: pd.DataFrame,
) -> None:
    """Scenario amount summary should preserve observations."""
    summary = scenario_amount_summary(transactions)

    assert summary["transaction_count"].sum() == 5


def test_behaviour_analysis_rejects_empty_dataframe() -> None:
    """Behavioural analysis should reject empty input."""
    with pytest.raises(
        ValueError,
        match="empty transaction dataset",
    ):
        amount_by_class(pd.DataFrame())