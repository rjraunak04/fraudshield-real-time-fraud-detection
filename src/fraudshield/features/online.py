"""Leakage-safe online feature engineering for FraudShield."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite

from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.state.models import EntityHistory
from fraudshield.state.store import OnlineStateStore
from fraudshield.streaming.events import TransactionEvent

# Day-03 frozen model contract is the single source of truth.
# The online serving layer must never maintain an independent copy.
ONLINE_MODEL_FEATURE_COLUMNS = tuple(MODEL_FEATURE_COLUMNS)


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Return a finite ratio with zero for unavailable history."""
    if denominator == 0.0:
        return 0.0

    value = numerator / denominator

    if not isfinite(value):
        return 0.0

    return float(value)


def _window_statistics(
    history: EntityHistory,
    *,
    event_time: datetime,
    window: timedelta,
) -> tuple[int, float]:
    """Return count and amount for a strict past-time window."""
    lower_bound = event_time - window

    transactions = (
        transaction
        for transaction in history.transactions
        if lower_bound <= transaction.event_time < event_time
    )

    count = 0
    amount = 0.0

    for transaction in transactions:
        count += 1
        amount += transaction.amount

    return count, float(amount)


def build_online_features(
    event: TransactionEvent,
    state: OnlineStateStore,
) -> dict[str, float]:
    """Build the frozen 21-feature model contract for one event."""
    customer_history = state.customer_history(
        event.customer_id,
        event.event_time,
    )
    terminal_history = state.terminal_history(
        event.terminal_id,
        event.event_time,
    )

    customer_count_1h, customer_amount_1h = _window_statistics(
        customer_history,
        event_time=event.event_time,
        window=timedelta(hours=1),
    )
    customer_count_24h, customer_amount_24h = _window_statistics(
        customer_history,
        event_time=event.event_time,
        window=timedelta(hours=24),
    )
    customer_count_7d, customer_amount_7d = _window_statistics(
        customer_history,
        event_time=event.event_time,
        window=timedelta(days=7),
    )

    customer_mean = customer_history.mean_amount
    terminal_mean = terminal_history.mean_amount

    hour = event.event_time.hour
    day_of_week = event.event_time.weekday()

    features: dict[str, float] = {
        "TX_AMOUNT": float(event.amount),
        "TX_HOUR": float(hour),
        "TX_DAY_OF_WEEK": float(day_of_week),
        "TX_DAY_OF_MONTH": float(event.event_time.day),
        "TX_MONTH": float(event.event_time.month),
        "TX_IS_WEEKEND": float(day_of_week >= 5),
        "TX_IS_NIGHT": float(hour < 6),
        "TX_IS_BUSINESS_HOURS": float(8 <= hour < 18),
        "CUSTOMER_TX_COUNT_1H": float(customer_count_1h),
        "CUSTOMER_TX_AMOUNT_1H": customer_amount_1h,
        "CUSTOMER_TX_COUNT_24H": float(customer_count_24h),
        "CUSTOMER_TX_AMOUNT_24H": customer_amount_24h,
        "CUSTOMER_TX_COUNT_7D": float(customer_count_7d),
        "CUSTOMER_TX_AMOUNT_7D": customer_amount_7d,
        "CUSTOMER_HIST_TX_COUNT": float(customer_history.count),
        "CUSTOMER_HIST_MEAN_AMOUNT": float(customer_mean),
        "CUSTOMER_HIST_MAX_AMOUNT": float(customer_history.max_amount),
        "CUSTOMER_AMOUNT_TO_HIST_MEAN": _safe_ratio(
            event.amount,
            customer_mean,
        ),
        "TERMINAL_HIST_TX_COUNT": float(terminal_history.count),
        "TERMINAL_HIST_MEAN_AMOUNT": float(terminal_mean),
        "TERMINAL_AMOUNT_TO_HIST_MEAN": _safe_ratio(
            event.amount,
            terminal_mean,
        ),
    }

    if tuple(features) != ONLINE_MODEL_FEATURE_COLUMNS:
        raise RuntimeError(
            "Online feature contract does not match frozen model feature order."
        )

    if len(features) != 21:
        raise RuntimeError(
            "Online feature contract must contain exactly 21 features."
        )

    if not all(isfinite(value) for value in features.values()):
        raise ValueError(
            "Online features contain non-finite values."
        )

    return features