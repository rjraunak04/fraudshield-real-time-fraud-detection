"""Verify real-time feature generation against the frozen model contract."""

from __future__ import annotations

from pathlib import Path

from fraudshield.features.online import ONLINE_MODEL_FEATURE_COLUMNS
from fraudshield.features.online_pipeline import OnlineFeaturePipeline
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.state.store import InMemoryStateStore
from fraudshield.streaming.simulator import TransactionStream

DATA_PATH = Path("data/raw/fraudshield_transactions.csv")
REPLAY_LIMIT = 1_000


def main() -> None:
    """Run Day-05 CP-03 real-time feature verification."""
    stream = TransactionStream.from_csv(DATA_PATH)
    state = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(state)

    replay = list(stream.iter_events(limit=REPLAY_LIMIT))

    if not replay:
        raise RuntimeError("Transaction replay is empty.")

    contract_matches = (
        ONLINE_MODEL_FEATURE_COLUMNS
        == tuple(MODEL_FEATURE_COLUMNS)
    )

    if not contract_matches:
        raise AssertionError(
            "Online feature contract differs from frozen model contract."
        )

    feature_count_violations = 0
    order_violations = 0
    state_violations = 0

    for item in replay:
        event = item.event

        customer_before = state.customer_history(
            event.customer_id,
            event.event_time,
        )
        terminal_before = state.terminal_history(
            event.terminal_id,
            event.event_time,
        )

        result = pipeline.transform(event)
        features = result.features

        if len(features) != len(MODEL_FEATURE_COLUMNS):
            feature_count_violations += 1

        if list(features) != MODEL_FEATURE_COLUMNS:
            order_violations += 1

        if (
            features["CUSTOMER_HIST_TX_COUNT"]
            != float(customer_before.count)
        ):
            state_violations += 1

        if (
            features["TERMINAL_HIST_TX_COUNT"]
            != float(terminal_before.count)
        ):
            state_violations += 1

        pipeline.commit(event)

    assert feature_count_violations == 0
    assert order_violations == 0
    assert state_violations == 0

    print()
    print(
        "FraudShield Day-05 CP-03 "
        "Real-Time Feature Verification"
    )
    print("=" * 72)
    print(f"Replay events: {len(replay):,}")
    print(
        "Frozen model features: "
        f"{len(MODEL_FEATURE_COLUMNS)}"
    )
    print(
        "Online features: "
        f"{len(ONLINE_MODEL_FEATURE_COLUMNS)}"
    )
    print(f"Contract parity: {contract_matches}")
    print(
        "Feature-count violations: "
        f"{feature_count_violations}"
    )
    print(f"Order violations: {order_violations}")
    print(
        "State/leakage violations: "
        f"{state_violations}"
    )
    print("Read-before-write semantics: PASSED")
    print("Locked test used: False")
    print("Model retrained: False")
    print("Threshold retuned: False")
    print()
    print("Day-05 CP-03 QA: PASSED")


if __name__ == "__main__":
    main()