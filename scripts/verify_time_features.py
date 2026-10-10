"""Verify FraudShield transaction-time feature engineering."""

from fraudshield.data.load import load_transactions
from fraudshield.features.time import (
    TIME_FEATURE_COLUMNS,
    add_transaction_time_features,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def main() -> None:
    """Generate and verify transaction-time features."""
    transactions = load_transactions(DATASET_PATH)

    featured = add_transaction_time_features(
        transactions
    )

    print()
    print("FraudShield Transaction-Time Features")
    print("=" * 70)

    print(
        featured[
            [
                "TRANSACTION_ID",
                "TX_DATETIME",
                *TIME_FEATURE_COLUMNS,
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Feature Integrity")
    print("=" * 70)

    print(
        "Input rows:",
        len(transactions),
    )

    print(
        "Output rows:",
        len(featured),
    )

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
        "Missing engineered values:",
        int(
            featured[
                TIME_FEATURE_COLUMNS
            ]
            .isna()
            .sum()
            .sum()
        ),
    )

    print()
    print("Feature Ranges")
    print("=" * 70)

    for column in TIME_FEATURE_COLUMNS:
        print(
            f"{column}: "
            f"min={featured[column].min()}, "
            f"max={featured[column].max()}"
        )


if __name__ == "__main__":
    main()