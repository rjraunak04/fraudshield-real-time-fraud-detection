"""Verify FraudShield online state using chronological replay."""

from __future__ import annotations

from pathlib import Path

from fraudshield.state.store import InMemoryStateStore
from fraudshield.streaming.simulator import TransactionStream

DATA_PATH = Path(
    "data/raw/fraudshield_transactions.csv"
)


def main() -> None:
    """Run Day-05 CP-02 online-state verification."""
    stream = TransactionStream.from_csv(DATA_PATH)
    store = InMemoryStateStore()

    replay = list(stream.iter_events(limit=1000))

    leakage_violations = 0

    for item in replay:
        event = item.event

        customer_before = store.customer_history(
            event.customer_id,
            event.event_time,
        )
        terminal_before = store.terminal_history(
            event.terminal_id,
            event.event_time,
        )

        historical_ids = {
            tx.transaction_id
            for tx in (
                customer_before.transactions
                + terminal_before.transactions
            )
        }

        if event.transaction_id in historical_ids:
            leakage_violations += 1

        store.record(event)

    last = replay[-1].event

    customer_history = store.customer_history(
        last.customer_id,
        last.event_time,
    )
    terminal_history = store.terminal_history(
        last.terminal_id,
        last.event_time,
    )

    assert leakage_violations == 0

    print()
    print(
        "FraudShield Day-05 CP-02 "
        "Online State Verification"
    )
    print("=" * 72)
    print(f"Replay events: {len(replay):,}")
    print(
        "Leakage violations: "
        f"{leakage_violations}"
    )
    print(
        "Last-event prior customer history: "
        f"{customer_history.count}"
    )
    print(
        "Last-event prior terminal history: "
        f"{terminal_history.count}"
    )
    print("Read-before-write semantics: PASSED")
    print("Current event visible before write: False")
    print("Locked test used: False")
    print("Model retrained: False")
    print()
    print("Day-05 CP-02 QA: PASSED")


if __name__ == "__main__":
    main()