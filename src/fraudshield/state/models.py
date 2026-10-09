"""Online state contracts for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class HistoricalTransaction:
    """Minimal historical transaction stored for online features."""

    transaction_id: int
    event_time: datetime
    amount: float


@dataclass(frozen=True)
class EntityHistory:
    """Past transaction history for one customer or terminal."""

    transactions: tuple[HistoricalTransaction, ...] = ()

    @property
    def count(self) -> int:
        """Return historical transaction count."""
        return len(self.transactions)

    @property
    def total_amount(self) -> float:
        """Return historical transaction amount."""
        return sum(item.amount for item in self.transactions)

    @property
    def mean_amount(self) -> float:
        """Return historical mean amount."""
        if not self.transactions:
            return 0.0
        return self.total_amount / self.count

    @property
    def max_amount(self) -> float:
        """Return historical maximum amount."""
        if not self.transactions:
            return 0.0
        return max(item.amount for item in self.transactions)