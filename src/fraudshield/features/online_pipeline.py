"""State-aware online feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from fraudshield.features.online import (
    ONLINE_MODEL_FEATURE_COLUMNS,
    build_online_features,
)
from fraudshield.state.store import OnlineStateStore
from fraudshield.streaming.events import TransactionEvent


@dataclass(frozen=True)
class OnlineFeatureResult:
    """Features produced for one incoming transaction."""

    transaction_id: int
    features: dict[str, float]

    def as_ordered_values(self) -> tuple[float, ...]:
        """Return values in frozen model feature order."""
        return tuple(
            self.features[column]
            for column in ONLINE_MODEL_FEATURE_COLUMNS
        )


class OnlineFeaturePipeline:
    """Build online features without leaking the current event."""

    def __init__(self, state: OnlineStateStore) -> None:
        self._state = state

    def transform(
        self,
        event: TransactionEvent,
    ) -> OnlineFeatureResult:
        """Read past state and build features without mutation."""
        features = build_online_features(
            event,
            self._state,
        )

        return OnlineFeatureResult(
            transaction_id=event.transaction_id,
            features=features,
        )

    def commit(
        self,
        event: TransactionEvent,
    ) -> None:
        """Write a completed event after feature computation."""
        self._state.record(event)