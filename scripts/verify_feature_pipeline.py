"""Run final Day-02 FraudShield feature engineering QA."""

import numpy as np

from fraudshield.data.load import load_transactions
from fraudshield.features.build import (
    ENGINEERED_FEATURE_COLUMNS,
    MODEL_EXCLUDED_COLUMNS,
    build_features,
    get_model_feature_columns,
    validate_feature_dataset,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def main() -> None:
    """Build and validate the complete feature dataset."""
    transactions = load_transactions(
        DATASET_PATH
    )

    featured = build_features(
        transactions
    )

    validate_feature_dataset(
        transactions,
        featured,
    )

    model_columns = get_model_feature_columns(
        featured
    )

    feature_values = featured[
        ENGINEERED_FEATURE_COLUMNS
    ].to_numpy(dtype="float64")

    print()
    print("FraudShield Day-02 Feature Pipeline QA")
    print("=" * 72)

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
        "Engineered feature count:",
        len(ENGINEERED_FEATURE_COLUMNS),
    )

    print(
        "Missing engineered values:",
        int(np.isnan(feature_values).sum()),
    )

    print(
        "Infinite engineered values:",
        int(np.isinf(feature_values).sum()),
    )

    print()
    print("Leakage Safeguards")
    print("=" * 72)

    for column in MODEL_EXCLUDED_COLUMNS:
        print(
            f"{column} excluded:",
            column not in model_columns,
        )

    print()
    print("Model Candidate Columns")
    print("=" * 72)

    for column in model_columns:
        print(column)

    print()
    print("Day-02 feature QA: PASSED")


if __name__ == "__main__":
    main()