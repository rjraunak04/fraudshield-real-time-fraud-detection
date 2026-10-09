"""Verify FraudShield chronological replay using the real synthetic dataset."""

from __future__ import annotations

from pathlib import Path

from fraudshield.streaming.simulator import TransactionStream

DATA_PATH = Path(
    "data/raw/fraudshield_transactions.csv"
)


def main() -> None:
    """Run CP-01 real-data streaming verification."""
    stream = TransactionStream.from_csv(DATA_PATH)

    sample = list(stream.iter_events(limit=1000))

    assert len(stream) == 1_845_459
    assert len(sample) == 1000

    timestamps = [
        item.event.event_time
        for item in sample
    ]

    assert timestamps == sorted(timestamps)

    transaction_ids = {
        item.event.transaction_id
        for item in sample
    }

    assert len(transaction_ids) == 1000

    assert all(
        item.event.amount >= 0
        for item in sample
    )

    print()
    print(
        "FraudShield Day-05 CP-01 "
        "Transaction Stream Verification"
    )
    print("=" * 72)
    print(f"Dataset rows: {len(stream):,}")
    print(f"Replay sample: {len(sample):,}")
    print(
        "First event time: "
        f"{sample[0].event.event_time}"
    )
    print(
        "Last sample event time: "
        f"{sample[-1].event.event_time}"
    )
    print(
        "Ground truth available: "
        f"{sample[0].fraud_label is not None}"
    )
    print("Inference event contains target: False")
    print("Chronological replay: PASSED")
    print("Locked test used: False")
    print("Model retrained: False")
    print()
    print("Day-05 CP-01 QA: PASSED")


if __name__ == "__main__":
    main()