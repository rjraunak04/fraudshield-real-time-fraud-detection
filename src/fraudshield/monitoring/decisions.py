"""Model-output and decision monitoring for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import Lock

from fraudshield.decision import (
    FROZEN_REVIEW_THRESHOLD,
    PROVISIONAL_BLOCK_THRESHOLD,
)


@dataclass(frozen=True, slots=True)
class ModelDecisionSnapshot:
    """Immutable model-output monitoring snapshot."""

    predictions_total: int

    approve_total: int
    review_total: int
    block_total: int

    probability_sum: float
    probability_min: float | None
    probability_max: float | None

    near_review_threshold_total: int
    near_block_threshold_total: int

    @property
    def approve_rate(self) -> float:
        """Return APPROVE fraction."""
        return self._rate(self.approve_total)

    @property
    def review_rate(self) -> float:
        """Return REVIEW fraction."""
        return self._rate(self.review_total)

    @property
    def block_rate(self) -> float:
        """Return BLOCK fraction."""
        return self._rate(self.block_total)

    @property
    def high_risk_rate(self) -> float:
        """Return REVIEW plus BLOCK fraction."""
        return self._rate(
            self.review_total + self.block_total
        )

    @property
    def mean_probability(self) -> float:
        """Return mean calibrated probability."""
        if self.predictions_total == 0:
            return 0.0

        return (
            self.probability_sum
            / self.predictions_total
        )

    @property
    def near_review_threshold_rate(self) -> float:
        """Return fraction close to review threshold."""
        return self._rate(
            self.near_review_threshold_total
        )

    @property
    def near_block_threshold_rate(self) -> float:
        """Return fraction close to block threshold."""
        return self._rate(
            self.near_block_threshold_total
        )

    def _rate(self, count: int) -> float:
        if self.predictions_total == 0:
            return 0.0

        return count / self.predictions_total


class ModelDecisionMonitor:
    """Thread-safe monitoring of model outputs and decisions."""

    def __init__(
        self,
        *,
        review_threshold: float = FROZEN_REVIEW_THRESHOLD,
        block_threshold: float = PROVISIONAL_BLOCK_THRESHOLD,
        threshold_margin: float = 0.005,
    ) -> None:
        if not 0.0 <= review_threshold <= 1.0:
            raise ValueError(
                "review_threshold must be between 0 and 1."
            )

        if not 0.0 <= block_threshold <= 1.0:
            raise ValueError(
                "block_threshold must be between 0 and 1."
            )

        if block_threshold <= review_threshold:
            raise ValueError(
                "block_threshold must exceed review_threshold."
            )

        if threshold_margin < 0.0:
            raise ValueError(
                "threshold_margin must be non-negative."
            )

        self._review_threshold = review_threshold
        self._block_threshold = block_threshold
        self._threshold_margin = threshold_margin

        self._lock = Lock()

        self._predictions_total = 0

        self._approve_total = 0
        self._review_total = 0
        self._block_total = 0

        self._probability_sum = 0.0
        self._probability_min: float | None = None
        self._probability_max: float | None = None

        self._near_review_threshold_total = 0
        self._near_block_threshold_total = 0

    def record(
        self,
        *,
        calibrated_probability: float,
        decision: str,
    ) -> None:
        """Record one model probability and final decision."""
        probability = float(
            calibrated_probability
        )

        if not isfinite(probability):
            raise ValueError(
                "calibrated_probability must be finite."
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "calibrated_probability must be "
                "between 0 and 1."
            )

        normalized_decision = decision.upper()

        if normalized_decision not in {
            "APPROVE",
            "REVIEW",
            "BLOCK",
        }:
            raise ValueError(
                f"Unsupported decision: {decision}."
            )

        with self._lock:
            self._predictions_total += 1

            if normalized_decision == "APPROVE":
                self._approve_total += 1
            elif normalized_decision == "REVIEW":
                self._review_total += 1
            else:
                self._block_total += 1

            self._probability_sum += probability

            if (
                self._probability_min is None
                or probability
                < self._probability_min
            ):
                self._probability_min = probability

            if (
                self._probability_max is None
                or probability
                > self._probability_max
            ):
                self._probability_max = probability

            if (
                abs(
                    probability
                    - self._review_threshold
                )
                <= self._threshold_margin
            ):
                self._near_review_threshold_total += 1

            if (
                abs(
                    probability
                    - self._block_threshold
                )
                <= self._threshold_margin
            ):
                self._near_block_threshold_total += 1

    def snapshot(self) -> ModelDecisionSnapshot:
        """Return immutable monitoring state."""
        with self._lock:
            return ModelDecisionSnapshot(
                predictions_total=(
                    self._predictions_total
                ),
                approve_total=self._approve_total,
                review_total=self._review_total,
                block_total=self._block_total,
                probability_sum=self._probability_sum,
                probability_min=self._probability_min,
                probability_max=self._probability_max,
                near_review_threshold_total=(
                    self._near_review_threshold_total
                ),
                near_block_threshold_total=(
                    self._near_block_threshold_total
                ),
            )