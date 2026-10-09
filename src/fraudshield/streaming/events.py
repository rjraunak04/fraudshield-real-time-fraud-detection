"""Raw transaction event contract for real-time FraudShield inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TransactionEvent:
    """Immutable raw transaction received by the online pipeline."""

    transaction_id: int
    event_time: datetime
    customer_id: int
    terminal_id: int
    amount: float

    def __post_init__(self) -> None:
        """Validate the online transaction contract."""
        if isinstance(self.transaction_id, bool) or not isinstance(
            self.transaction_id, int
        ):
            raise TypeError("transaction_id must be an integer.")

        if self.transaction_id < 0:
            raise ValueError("transaction_id must be non-negative.")

        if not isinstance(self.event_time, datetime):
            raise TypeError("event_time must be a datetime.")

        if isinstance(self.customer_id, bool) or not isinstance(
            self.customer_id, int
        ):
            raise TypeError("customer_id must be an integer.")

        if self.customer_id < 0:
            raise ValueError("customer_id must be non-negative.")

        if isinstance(self.terminal_id, bool) or not isinstance(
            self.terminal_id, int
        ):
            raise TypeError("terminal_id must be an integer.")

        if self.terminal_id < 0:
            raise ValueError("terminal_id must be non-negative.")

        if isinstance(self.amount, bool) or not isinstance(
            self.amount, (int, float)
        ):
            raise TypeError("amount must be numeric.")

        amount = float(self.amount)

        if not math.isfinite(amount):
            raise ValueError("amount must be finite.")

        if amount < 0:
            raise ValueError("amount must be non-negative.")


@dataclass(frozen=True)
class SimulatedTransaction:
    """Transaction event plus simulator-only ground truth metadata."""

    event: TransactionEvent
    fraud_label: int | None = None
    fraud_scenario: int | None = None

    def __post_init__(self) -> None:
        """Validate optional simulator metadata."""
        if self.fraud_label is not None and self.fraud_label not in (0, 1):
            raise ValueError("fraud_label must be 0, 1, or None.")

        if self.fraud_scenario is not None:
            if isinstance(self.fraud_scenario, bool) or not isinstance(
                self.fraud_scenario, int
            ):
                raise TypeError("fraud_scenario must be an integer or None.")

            if self.fraud_scenario < 0:
                raise ValueError("fraud_scenario must be non-negative.")