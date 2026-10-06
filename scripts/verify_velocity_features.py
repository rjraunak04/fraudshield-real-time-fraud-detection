"""Verify FraudShield customer velocity features."""

from fraudshield.data.load import load_transactions
from fraudshield.features.velocity import (
    VELOCITY_FEATURE_COLUMNS,
    add_customer_velocity_features,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def main() -> None:
    """Generate and verify customer velocity features."""
    transactions = load_transactions(DATASET_PATH)

    featured = add_customer_velocity_features(
        transactions
    )

    print()
    print("FraudShield Customer Velocity Features")
    print("=" * 72)

    print(
        featured[
            [
                "TRANSACTION_ID",
                "TX_DATETIME",
                "CUSTOMER_ID",
                "TX_AMOUNT",
                *VELOCITY_FEATURE_COLUMNS,
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("Integrity Checks")
    print("=" * 72)

    print("Input rows:", len(transactions))
    print("Output rows:", len(featured))

    print(
        "Rows preserved:",
        len(transactions) == len(featured),
    )

    print(
        "Transaction IDs preserved:",
        transactions["TRANSACTION_ID"].equals(
            featured["TRANSACTION_ID"]
        ),
    )

    print(
        "Missing velocity values:",
        int(
            featured[
                VELOCITY_FEATURE_COLUMNS
            ]
            .isna()
            .sum()
            .sum()
        ),
    )

    count_columns = [
        column
        for column in VELOCITY_FEATURE_COLUMNS
        if "_COUNT_" in column
    ]

    amount_columns = [
        column
        for column in VELOCITY_FEATURE_COLUMNS
        if "_AMOUNT_" in column
    ]

    print(
        "Negative velocity counts:",
        int(
            (featured[count_columns] < 0)
            .sum()
            .sum()
        ),
    )

    print(
        "Negative velocity amounts:",
        int(
            (featured[amount_columns] < -1e-9)
            .sum()
            .sum()
        ),
    )

    print()
    print("Velocity Feature Summary")
    print("=" * 72)

    print(
        featured[
            VELOCITY_FEATURE_COLUMNS
        ]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()