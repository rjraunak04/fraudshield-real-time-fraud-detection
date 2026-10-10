"""Tests for FraudShield transaction streaming."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from fraudshield.streaming.events import (
    SimulatedTransaction,
    TransactionEvent,
)
from fraudshield.streaming.simulator import TransactionStream


def sample_frame() -> pd.DataFrame:
    """Create intentionally unordered transaction data."""
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [3, 1, 2],
            "TX_DATETIME": [
                "2018-04-01 10:05:00",
                "2018-04-01 10:00:00",
                "2018-04-01 10:00:00",
            ],
            "CUSTOMER_ID": [30, 10, 20],
            "TERMINAL_ID": [300, 100, 200],
            "TX_AMOUNT": [30.0, 10.0, 20.0],
            "TX_FRAUD": [0, 1, 0],
            "TX_FRAUD_SCENARIO": [0, 2, 0],
        }
    )


def test_transaction_event_valid() -> None:
    event = TransactionEvent(
        transaction_id=1,
        event_time=datetime(2018, 4, 1),
        customer_id=10,
        terminal_id=20,
        amount=25.0,
    )

    assert event.amount == 25.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_id", -1),
        ("customer_id", -1),
        ("terminal_id", -1),
        ("amount", -0.01),
    ],
)
def test_transaction_event_rejects_negative_values(
    field: str,
    value: int | float,
) -> None:
    kwargs = {
        "transaction_id": 1,
        "event_time": datetime(2018, 4, 1),
        "customer_id": 10,
        "terminal_id": 20,
        "amount": 25.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        TransactionEvent(**kwargs)


@pytest.mark.parametrize(
    "amount",
    [float("nan"), float("inf"), float("-inf")],
)
def test_transaction_event_rejects_nonfinite_amount(
    amount: float,
) -> None:
    with pytest.raises(ValueError):
        TransactionEvent(
            transaction_id=1,
            event_time=datetime(2018, 4, 1),
            customer_id=10,
            terminal_id=20,
            amount=amount,
        )


def test_stream_orders_chronologically() -> None:
    stream = TransactionStream(sample_frame())

    events = list(stream.iter_events())

    assert [
        item.event.transaction_id
        for item in events
    ] == [1, 2, 3]


def test_stream_preserves_ground_truth_separately() -> None:
    stream = TransactionStream(sample_frame())

    first = next(stream.iter_events())

    assert first.fraud_label == 1
    assert first.fraud_scenario == 2

    assert not hasattr(
        first.event,
        "fraud_label",
    )


def test_stream_supports_missing_ground_truth() -> None:
    frame = sample_frame().drop(
        columns=[
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
        ]
    )

    first = next(
        TransactionStream(frame).iter_events()
    )

    assert first.fraud_label is None
    assert first.fraud_scenario is None


def test_stream_limit() -> None:
    stream = TransactionStream(sample_frame())

    events = list(stream.iter_events(limit=2))

    assert len(events) == 2


def test_stream_rejects_negative_limit() -> None:
    stream = TransactionStream(sample_frame())

    with pytest.raises(ValueError):
        list(stream.iter_events(limit=-1))


def test_stream_rejects_missing_columns() -> None:
    frame = sample_frame().drop(
        columns=["TX_AMOUNT"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required streaming columns",
    ):
        TransactionStream(frame)


def test_stream_rejects_duplicate_transaction_ids() -> None:
    frame = sample_frame()
    frame.loc[1, "TRANSACTION_ID"] = 3

    with pytest.raises(
        ValueError,
        match="TRANSACTION_ID must be unique",
    ):
        TransactionStream(frame)


def test_simulated_transaction_rejects_bad_label() -> None:
    event = TransactionEvent(
        transaction_id=1,
        event_time=datetime(2018, 4, 1),
        customer_id=10,
        terminal_id=20,
        amount=25.0,
    )

    with pytest.raises(ValueError):
        SimulatedTransaction(
            event=event,
            fraud_label=2,
        )