"""Redis-backed online transaction state."""

from __future__ import annotations

import json
from datetime import datetime

from redis import Redis

from fraudshield.state.models import (
    EntityHistory,
    HistoricalTransaction,
)
from fraudshield.streaming.events import TransactionEvent


class RedisStateStore:
    """Redis implementation of the online feature state contract."""

    def __init__(
        self,
        client: Redis,
        *,
        namespace: str = "fraudshield",
    ) -> None:
        if not namespace.strip():
            raise ValueError(
                "namespace must not be empty."
            )

        self._client = client
        self._namespace = namespace

    def _customer_key(self, customer_id: int) -> str:
        return (
            f"{self._namespace}:customer:"
            f"{customer_id}:transactions"
        )

    def _terminal_key(self, terminal_id: int) -> str:
        return (
            f"{self._namespace}:terminal:"
            f"{terminal_id}:transactions"
        )

    def _transaction_key(
        self,
        transaction_id: int,
    ) -> str:
        return (
            f"{self._namespace}:transaction:"
            f"{transaction_id}"
        )

    @staticmethod
    def _serialize(
        event: TransactionEvent,
    ) -> str:
        return json.dumps(
            {
                "transaction_id": event.transaction_id,
                "event_time": event.event_time.isoformat(),
                "amount": event.amount,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _deserialize(
        payload: bytes | str,
    ) -> HistoricalTransaction:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        data = json.loads(payload)

        return HistoricalTransaction(
            transaction_id=int(
                data["transaction_id"]
            ),
            event_time=datetime.fromisoformat(
                data["event_time"]
            ),
            amount=float(data["amount"]),
        )

    def _history(
        self,
        key: str,
        before: datetime,
    ) -> EntityHistory:
        # Exclusive upper bound prevents current/same-time
        # transactions from entering the feature state.
        upper = f"({before.timestamp()}"

        payloads = self._client.zrangebyscore(
            key,
            "-inf",
            upper,
        )

        return EntityHistory(
            transactions=tuple(
                self._deserialize(payload)
                for payload in payloads
            )
        )

    def customer_history(
        self,
        customer_id: int,
        before: datetime,
    ) -> EntityHistory:
        """Read strictly past customer history."""
        return self._history(
            self._customer_key(customer_id),
            before,
        )

    def terminal_history(
        self,
        terminal_id: int,
        before: datetime,
    ) -> EntityHistory:
        """Read strictly past terminal history."""
        return self._history(
            self._terminal_key(terminal_id),
            before,
        )

    def record(self, event: TransactionEvent) -> None:
        """Atomically register event and update entity histories."""
        marker = self._transaction_key(
            event.transaction_id
        )

        created = self._client.set(
            marker,
            "1",
            nx=True,
        )

        if not created:
            raise ValueError(
                "Transaction has already been recorded."
            )

        payload = self._serialize(event)
        score = event.event_time.timestamp()

        try:
            with self._client.pipeline(
                transaction=True
            ) as pipeline:
                pipeline.zadd(
                    self._customer_key(
                        event.customer_id
                    ),
                    {payload: score},
                )
                pipeline.zadd(
                    self._terminal_key(
                        event.terminal_id
                    ),
                    {payload: score},
                )
                pipeline.execute()
        except Exception:
            self._client.delete(marker)
            raise