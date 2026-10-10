"""Verify FraudShield temporal train/validation/test partitions."""

from fraudshield.data.load import load_transactions
from fraudshield.data.split import (
    split_summary,
    temporal_train_validation_test_split,
    validate_temporal_split,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def main() -> None:
    """Create and validate chronological FraudShield partitions."""
    transactions = load_transactions(DATASET_PATH)

    split = temporal_train_validation_test_split(
        transactions,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    validate_temporal_split(split)

    summary = split_summary(split)

    print()
    print("FraudShield Temporal Split")
    print("=" * 70)
    print(summary.to_string(index=False))

    original_ids = set(
        transactions["TRANSACTION_ID"]
    )

    combined_ids = (
        set(split.train["TRANSACTION_ID"])
        | set(split.validation["TRANSACTION_ID"])
        | set(split.test["TRANSACTION_ID"])
    )

    print()
    print("Integrity Checks")
    print("=" * 70)
    print(
        "Original transactions:",
        len(transactions),
    )
    print(
        "Partitioned transactions:",
        (
            len(split.train)
            + len(split.validation)
            + len(split.test)
        ),
    )
    print(
        "All transaction IDs preserved:",
        original_ids == combined_ids,
    )
    print("Temporal validation: PASSED")


if __name__ == "__main__":
    main()