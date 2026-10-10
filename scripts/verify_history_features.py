"""Verify FraudShield historical behavioural features."""

import numpy as np

from fraudshield.data.load import load_transactions
from fraudshield.features.history import (
    HISTORICAL_FEATURE_COLUMNS,
    add_historical_behaviour_features,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def main() -> None:
    """Generate and verify historical behavioural features."""
    transactions = load_transactions(DATASET_PATH)

    featured = add_historical_behaviour_features(
        transactions
    )

    print()
    print("FraudShield Historical Behaviour Features")
    print("=" * 72)

    print(
        featured[
            [
                "TRANSACTION_ID",
                "TX_DATETIME",
                "CUSTOMER_ID",
                "TERMINAL_ID",
                "TX_AMOUNT",
                *HISTORICAL_FEATURE_COLUMNS,
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
        "Missing historical values:",
        int(
            featured[
                HISTORICAL_FEATURE_COLUMNS
            ]
            .isna()
            .sum()
            .sum()
        ),
    )

    values = featured[
        HISTORICAL_FEATURE_COLUMNS
    ].to_numpy(dtype="float64")

    print(
        "Infinite historical values:",
        int(np.isinf(values).sum()),
    )

    count_columns = [
        "CUSTOMER_HIST_TX_COUNT",
        "TERMINAL_HIST_TX_COUNT",
    ]

    print(
        "Negative historical counts:",
        int(
            (featured[count_columns] < 0)
            .sum()
            .sum()
        ),
    )

    print()
    print("Historical Feature Summary")
    print("=" * 72)

    print(
        featured[
            HISTORICAL_FEATURE_COLUMNS
        ]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()