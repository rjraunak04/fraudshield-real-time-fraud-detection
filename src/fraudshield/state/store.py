"""State-store abstraction and deterministic in-memory backend."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from fraudshield.state.models import (
    EntityHistory,
    HistoricalTransaction,
)
from fraudshield.streaming.events import TransactionEvent


class OnlineStateStore(Protocol):
    """Storage contract required by online feature engineering."""

    def customer_history(
        self,
        customer_id: int,
        before: datetime,
    ) -> EntityHistory:
        """Return customer transactions strictly before event time."""
        ...

    def terminal_history(
        self,
        terminal_id: int,
        before: datetime,
    ) -> EntityHistory:
        """Return terminal transactions strictly before event time."""
        ...

    def record(self, event: TransactionEvent) -> None:
        """Record a completed transaction."""
        ...


class InMemoryStateStore:
    """Deterministic state backend used for tests and local QA."""

    def __init__(self) -> None:
        self._customers: dict[
            int, list[HistoricalTransaction]
        ] = defaultdict(list)
        self._terminals: dict[
            int, list[HistoricalTransaction]
        ] = defaultdict(list)
        self._transaction_ids: set[int] = set()

    @staticmethod
    def _history_before(
        history: Sequence[HistoricalTransaction],
        before: datetime,
    ) -> EntityHistory:
        return EntityHistory(
            transactions=tuple(
                item
                for item in history
                if item.event_time < before
            )
        )

    def customer_history(
        self,
        customer_id: int,
        before: datetime,
    ) -> EntityHistory:
        """Return strictly past customer state."""
        return self._history_before(
            self._customers[customer_id],
            before,
        )

    def terminal_history(
        self,
        terminal_id: int,
        before: datetime,
    ) -> EntityHistory:
        """Return strictly past terminal state."""
        return self._history_before(
            self._terminals[terminal_id],
            before,
        )

    def record(self, event: TransactionEvent) -> None:
        """Persist one event into customer and terminal state."""
        if event.transaction_id in self._transaction_ids:
            raise ValueError(
                "Transaction has already been recorded."
            )

        historical = HistoricalTransaction(
            transaction_id=event.transaction_id,
            event_time=event.event_time,
            amount=event.amount,
        )

        self._customers[event.customer_id].append(
            historical
        )
        self._terminals[event.terminal_id].append(
            historical
        )
        self._transaction_ids.add(event.transaction_id)