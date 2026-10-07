"""Verify the FraudShield Day-03 modeling dataset contract."""

from fraudshield.data.load import load_transactions
from fraudshield.modeling.dataset import (
    FORBIDDEN_MODEL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    build_modeling_dataset,
    modeling_dataset_summary,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def main() -> None:
    """Build and verify chronological model-ready datasets."""
    transactions = load_transactions(
        DATASET_PATH
    )

    dataset = build_modeling_dataset(
        transactions
    )

    summary = modeling_dataset_summary(
        dataset
    )

    print()
    print("FraudShield Day-03 Modeling Dataset")
    print("=" * 72)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("Feature Contract")
    print("=" * 72)

    print(
        "Approved model feature count:",
        len(MODEL_FEATURE_COLUMNS),
    )

    for feature in MODEL_FEATURE_COLUMNS:
        print(feature)

    print()
    print("Leakage Safeguards")
    print("=" * 72)

    for column in sorted(
        FORBIDDEN_MODEL_COLUMNS
    ):
        print(
            f"{column} excluded:",
            column not in dataset.train.X.columns,
        )

    total_rows = (
        len(dataset.train.X)
        + len(dataset.validation.X)
        + len(dataset.test.X)
    )

    print()
    print("Integrity Checks")
    print("=" * 72)

    print(
        "Original rows:",
        len(transactions),
    )

    print(
        "Partitioned rows:",
        total_rows,
    )

    print(
        "Rows preserved:",
        total_rows == len(transactions),
    )

    print(
        "Train feature shape:",
        dataset.train.X.shape,
    )

    print(
        "Validation feature shape:",
        dataset.validation.X.shape,
    )

    print(
        "Locked test feature shape:",
        dataset.test.X.shape,
    )

    print()
    print("CP-01 modeling dataset QA: PASSED")


if __name__ == "__main__":
    main()