"""Tests for leakage-safe online feature engineering."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fraudshield.features.online import (
    ONLINE_MODEL_FEATURE_COLUMNS,
    build_online_features,
)
from fraudshield.features.online_pipeline import OnlineFeaturePipeline
from fraudshield.state.store import InMemoryStateStore
from fraudshield.streaming.events import TransactionEvent

BASE_TIME = datetime(2018, 4, 2, 12, 0)


def make_event(
    transaction_id: int,
    *,
    event_time: datetime = BASE_TIME,
    customer_id: int = 10,
    terminal_id: int = 20,
    amount: float = 100.0,
) -> TransactionEvent:
    """Create a deterministic transaction event."""
    return TransactionEvent(
        transaction_id=transaction_id,
        event_time=event_time,
        customer_id=customer_id,
        terminal_id=terminal_id,
        amount=amount,
    )


def test_feature_contract_contains_21_features() -> None:
    assert len(ONLINE_MODEL_FEATURE_COLUMNS) == 21


def test_new_entity_has_zero_history() -> None:
    store = InMemoryStateStore()
    event = make_event(1)

    features = build_online_features(event, store)

    assert features["CUSTOMER_HIST_TX_COUNT"] == 0.0
    assert features["CUSTOMER_HIST_MEAN_AMOUNT"] == 0.0
    assert features["CUSTOMER_HIST_MAX_AMOUNT"] == 0.0
    assert features["CUSTOMER_AMOUNT_TO_HIST_MEAN"] == 0.0
    assert features["TERMINAL_HIST_TX_COUNT"] == 0.0
    assert features["TERMINAL_HIST_MEAN_AMOUNT"] == 0.0
    assert features["TERMINAL_AMOUNT_TO_HIST_MEAN"] == 0.0


def test_current_event_is_not_in_its_features() -> None:
    store = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(store)
    event = make_event(1)

    result = pipeline.transform(event)

    assert result.features["CUSTOMER_HIST_TX_COUNT"] == 0.0
    assert result.features["TERMINAL_HIST_TX_COUNT"] == 0.0


def test_transform_does_not_mutate_state() -> None:
    store = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(store)
    event = make_event(1)

    pipeline.transform(event)
    pipeline.transform(event)

    assert (
        store.customer_history(
            event.customer_id,
            event.event_time + timedelta(seconds=1),
        ).count
        == 0
    )


def test_commit_makes_event_available_to_future_event() -> None:
    store = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(store)

    first = make_event(1, amount=40.0)
    second = make_event(
        2,
        event_time=BASE_TIME + timedelta(minutes=10),
        amount=80.0,
    )

    pipeline.transform(first)
    pipeline.commit(first)

    result = pipeline.transform(second)

    assert result.features["CUSTOMER_HIST_TX_COUNT"] == 1.0
    assert result.features["CUSTOMER_HIST_MEAN_AMOUNT"] == 40.0
    assert result.features["CUSTOMER_HIST_MAX_AMOUNT"] == 40.0
    assert result.features["CUSTOMER_AMOUNT_TO_HIST_MEAN"] == 2.0


def test_customer_velocity_windows() -> None:
    store = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(store)

    events = [
        make_event(
            1,
            event_time=BASE_TIME - timedelta(minutes=30),
            amount=10.0,
        ),
        make_event(
            2,
            event_time=BASE_TIME - timedelta(hours=2),
            amount=20.0,
        ),
        make_event(
            3,
            event_time=BASE_TIME - timedelta(days=2),
            amount=30.0,
        ),
        make_event(
            4,
            event_time=BASE_TIME - timedelta(days=8),
            amount=40.0,
        ),
    ]

    for historical_event in events:
        pipeline.commit(historical_event)

    current = make_event(5)
    features = pipeline.transform(current).features

    assert features["CUSTOMER_TX_COUNT_1H"] == 1.0
    assert features["CUSTOMER_TX_AMOUNT_1H"] == 10.0

    assert features["CUSTOMER_TX_COUNT_24H"] == 2.0
    assert features["CUSTOMER_TX_AMOUNT_24H"] == 30.0

    assert features["CUSTOMER_TX_COUNT_7D"] == 3.0
    assert features["CUSTOMER_TX_AMOUNT_7D"] == 60.0

    assert features["CUSTOMER_HIST_TX_COUNT"] == 4.0


def test_velocity_window_includes_exact_lower_boundary() -> None:
    store = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(store)

    historical = make_event(
        1,
        event_time=BASE_TIME - timedelta(hours=1),
        amount=25.0,
    )
    pipeline.commit(historical)

    features = pipeline.transform(make_event(2)).features

    assert features["CUSTOMER_TX_COUNT_1H"] == 1.0
    assert features["CUSTOMER_TX_AMOUNT_1H"] == 25.0


def test_same_timestamp_history_is_excluded() -> None:
    store = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(store)

    first = make_event(1, amount=20.0)
    pipeline.commit(first)

    second = make_event(2, amount=30.0)
    features = pipeline.transform(second).features

    assert features["CUSTOMER_TX_COUNT_1H"] == 0.0
    assert features["CUSTOMER_HIST_TX_COUNT"] == 0.0


def test_terminal_history_is_independent_from_customer_history() -> None:
    store = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(store)

    previous = make_event(
        1,
        customer_id=99,
        terminal_id=20,
        amount=50.0,
    )
    pipeline.commit(previous)

    current = make_event(
        2,
        customer_id=10,
        terminal_id=20,
        amount=100.0,
        event_time=BASE_TIME + timedelta(minutes=1),
    )

    features = pipeline.transform(current).features

    assert features["CUSTOMER_HIST_TX_COUNT"] == 0.0
    assert features["TERMINAL_HIST_TX_COUNT"] == 1.0
    assert features["TERMINAL_HIST_MEAN_AMOUNT"] == 50.0
    assert features["TERMINAL_AMOUNT_TO_HIST_MEAN"] == 2.0


def test_time_features() -> None:
    store = InMemoryStateStore()

    saturday_night = make_event(
        1,
        event_time=datetime(2018, 4, 7, 2, 30),
    )

    features = build_online_features(
        saturday_night,
        store,
    )

    assert features["TX_HOUR"] == 2.0
    assert features["TX_DAY_OF_WEEK"] == 5.0
    assert features["TX_DAY_OF_MONTH"] == 7.0
    assert features["TX_MONTH"] == 4.0
    assert features["TX_IS_WEEKEND"] == 1.0
    assert features["TX_IS_NIGHT"] == 1.0
    assert features["TX_IS_BUSINESS_HOURS"] == 0.0


def test_feature_order_matches_contract() -> None:
    store = InMemoryStateStore()

    features = build_online_features(
        make_event(1),
        store,
    )

    assert tuple(features) == ONLINE_MODEL_FEATURE_COLUMNS


def test_ordered_values_contains_all_features() -> None:
    store = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(store)

    result = pipeline.transform(make_event(1))

    assert len(result.as_ordered_values()) == 21


def test_duplicate_commit_is_rejected() -> None:
    store = InMemoryStateStore()
    pipeline = OnlineFeaturePipeline(store)
    event = make_event(1)

    pipeline.commit(event)

    with pytest.raises(
        ValueError,
        match="already been recorded",
    ):
        pipeline.commit(event)
def test_online_contract_matches_frozen_model_contract() -> None:
    from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS

    assert ONLINE_MODEL_FEATURE_COLUMNS == tuple(MODEL_FEATURE_COLUMNS)


def test_online_features_match_frozen_model_feature_order() -> None:
    from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS

    store = InMemoryStateStore()

    features = build_online_features(
        make_event(1),
        store,
    )

    assert list(features) == MODEL_FEATURE_COLUMNS