"""Run target and temporal exploratory analysis for FraudShield."""

from fraudshield.analysis.temporal import (
    daily_fraud_summary,
    fraud_scenario_summary,
    hourly_fraud_summary,
    monthly_fraud_summary,
    target_summary,
    temporal_quality_summary,
)
from fraudshield.data.load import load_transactions

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def print_section(title: str) -> None:
    """Print a readable analysis section heading."""
    print()
    print(title)
    print("=" * len(title))


def main() -> None:
    """Run FraudShield target and temporal analysis."""
    transactions = load_transactions(DATASET_PATH)

    print_section("Target Distribution")
    print(
        target_summary(transactions).to_string(
            index=False
        )
    )

    print_section("Temporal Quality")
    quality = temporal_quality_summary(transactions)

    for metric, value in quality.items():
        print(f"{metric}: {value}")

    print_section("Monthly Fraud Summary")
    print(
        monthly_fraud_summary(transactions).to_string(
            index=False
        )
    )

    print_section("Hourly Fraud Summary")
    print(
        hourly_fraud_summary(transactions).to_string(
            index=False
        )
    )

    print_section("Fraud Scenario Summary")
    print(
        fraud_scenario_summary(transactions).to_string(
            index=False
        )
    )

    daily = daily_fraud_summary(transactions)

    print_section("Daily Fraud Summary Statistics")
    print(
        daily[
            [
                "transaction_count",
                "fraud_count",
                "fraud_rate",
            ]
        ]
        .describe()
        .to_string()
    )

    highest_fraud_days = (
        daily.sort_values(
            ["fraud_rate", "fraud_count"],
            ascending=False,
        )
        .head(10)
    )

    print_section("Top 10 Days by Fraud Rate")
    print(
        highest_fraud_days.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()