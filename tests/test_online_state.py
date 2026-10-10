"""Tests for FraudShield online transaction state."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fraudshield.state.store import InMemoryStateStore
from fraudshield.streaming.events import TransactionEvent

BASE_TIME = datetime(2018, 4, 1, 10, 0)


def event(
    transaction_id: int,
    *,
    minutes: int = 0,
    customer_id: int = 10,
    terminal_id: int = 20,
    amount: float = 25.0,
) -> TransactionEvent:
    """Build a deterministic transaction event."""
    return TransactionEvent(
        transaction_id=transaction_id,
        event_time=BASE_TIME
        + timedelta(minutes=minutes),
        customer_id=customer_id,
        terminal_id=terminal_id,
        amount=amount,
    )


def test_empty_customer_history() -> None:
    store = InMemoryStateStore()

    history = store.customer_history(
        10,
        BASE_TIME,
    )

    assert history.count == 0
    assert history.total_amount == 0.0
    assert history.mean_amount == 0.0
    assert history.max_amount == 0.0


def test_record_populates_customer_history() -> None:
    store = InMemoryStateStore()
    first = event(1)

    store.record(first)

    history = store.customer_history(
        10,
        BASE_TIME + timedelta(minutes=1),
    )

    assert history.count == 1
    assert history.total_amount == 25.0


def test_record_populates_terminal_history() -> None:
    store = InMemoryStateStore()
    store.record(event(1))

    history = store.terminal_history(
        20,
        BASE_TIME + timedelta(minutes=1),
    )

    assert history.count == 1


def test_current_transaction_is_not_visible() -> None:
    store = InMemoryStateStore()
    store.record(event(1))

    history = store.customer_history(
        10,
        BASE_TIME,
    )

    assert history.count == 0


def test_same_timestamp_transactions_are_not_visible() -> None:
    store = InMemoryStateStore()
    store.record(event(1))
    store.record(event(2))

    history = store.customer_history(
        10,
        BASE_TIME,
    )

    assert history.count == 0


def test_future_transaction_is_not_visible() -> None:
    store = InMemoryStateStore()
    store.record(event(1, minutes=10))

    history = store.customer_history(
        10,
        BASE_TIME + timedelta(minutes=5),
    )

    assert history.count == 0


def test_history_statistics() -> None:
    store = InMemoryStateStore()

    store.record(event(1, amount=10.0))
    store.record(
        event(
            2,
            minutes=1,
            amount=30.0,
        )
    )

    history = store.customer_history(
        10,
        BASE_TIME + timedelta(minutes=2),
    )

    assert history.count == 2
    assert history.total_amount == 40.0
    assert history.mean_amount == 20.0
    assert history.max_amount == 30.0


def test_customer_state_is_isolated() -> None:
    store = InMemoryStateStore()

    store.record(event(1, customer_id=10))
    store.record(event(2, customer_id=11))

    history = store.customer_history(
        10,
        BASE_TIME + timedelta(minutes=1),
    )

    assert history.count == 1
    assert (
        history.transactions[0].transaction_id
        == 1
    )


def test_terminal_state_is_isolated() -> None:
    store = InMemoryStateStore()

    store.record(event(1, terminal_id=20))
    store.record(event(2, terminal_id=21))

    history = store.terminal_history(
        20,
        BASE_TIME + timedelta(minutes=1),
    )

    assert history.count == 1
    assert (
        history.transactions[0].transaction_id
        == 1
    )


def test_duplicate_transaction_rejected() -> None:
    store = InMemoryStateStore()
    first = event(1)

    store.record(first)

    with pytest.raises(
        ValueError,
        match="already been recorded",
    ):
        store.record(first)


def test_history_does_not_cross_entities() -> None:
    store = InMemoryStateStore()

    store.record(
        event(
            1,
            customer_id=10,
            terminal_id=20,
        )
    )

    assert (
        store.customer_history(
            99,
            BASE_TIME + timedelta(minutes=1),
        ).count
        == 0
    )

    assert (
        store.terminal_history(
            99,
            BASE_TIME + timedelta(minutes=1),
        ).count
        == 0
    )