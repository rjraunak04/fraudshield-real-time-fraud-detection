"""Profile the local FraudShield transaction dataset."""

from fraudshield.data.load import load_transactions
from fraudshield.data.profile import (
    build_dataset_profile,
    profile_to_frame,
)

DATASET_PATH = "data/raw/fraudshield_transactions.csv"


def main() -> None:
    """Load, validate, and profile the FraudShield dataset."""
    transactions = load_transactions(DATASET_PATH)

    profile = build_dataset_profile(transactions)

    print("\nFraudShield Dataset Profile")
    print("=" * 60)

    print(
        profile_to_frame(profile).to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()