"""Run behavioural exploratory analysis for FraudShield."""

from fraudshield.analysis.behaviour import (
    amount_band_summary,
    amount_by_class,
    amount_quantile_summary,
    customer_summary,
    scenario_amount_summary,
    terminal_summary,
)
from fraudshield.data.load import load_transactions

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def print_section(title: str) -> None:
    """Print a readable section heading."""
    print()
    print(title)
    print("=" * len(title))


def main() -> None:
    """Run amount, customer, and terminal behavioural analysis."""
    transactions = load_transactions(DATASET_PATH)

    print_section("Transaction Amount by Fraud Class")
    print(
        amount_by_class(transactions).to_string(
            index=False
        )
    )

    print_section("Amount Quantiles by Fraud Class")
    print(
        amount_quantile_summary(transactions).to_string(
            index=False
        )
    )

    print_section("Fraud by Transaction Amount Band")
    print(
        amount_band_summary(transactions).to_string(
            index=False
        )
    )

    customers = customer_summary(transactions)

    print_section("Customer Behaviour Distribution")
    print(
        customers[
            [
                "transaction_count",
                "fraud_count",
                "total_amount",
                "mean_amount",
                "unique_terminals",
                "fraud_rate",
            ]
        ]
        .describe()
        .to_string()
    )

    print_section("Top Customers by Fraud Count")
    print(
        customers.sort_values(
            ["fraud_count", "transaction_count"],
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    terminals = terminal_summary(transactions)

    print_section("Terminal Behaviour Distribution")
    print(
        terminals[
            [
                "transaction_count",
                "fraud_count",
                "total_amount",
                "mean_amount",
                "unique_customers",
                "fraud_rate",
            ]
        ]
        .describe()
        .to_string()
    )

    print_section("Top Terminals by Fraud Count")
    print(
        terminals.sort_values(
            ["fraud_count", "transaction_count"],
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print_section("Amount by Simulator Scenario")
    print(
        scenario_amount_summary(transactions).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()