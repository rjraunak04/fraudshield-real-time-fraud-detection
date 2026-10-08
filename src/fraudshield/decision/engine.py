"""Risk decision engine for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from fraudshield.explainability.reason_codes import ReasonCode
from fraudshield.scoring.service import FraudScore

FROZEN_REVIEW_THRESHOLD = 0.048
PROVISIONAL_BLOCK_THRESHOLD = 0.100
DECISION_POLICY_VERSION = "day-04-three-way-policy-v1"


class DecisionAction(StrEnum):
    """Supported FraudShield transaction actions."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class DecisionPolicy:
    """Configurable three-way fraud decision policy."""

    review_threshold: float = FROZEN_REVIEW_THRESHOLD
    block_threshold: float = PROVISIONAL_BLOCK_THRESHOLD
    version: str = DECISION_POLICY_VERSION

    def __post_init__(self) -> None:
        """Validate policy configuration."""
        _validate_probability(
            self.review_threshold,
            name="review_threshold",
        )
        _validate_probability(
            self.block_threshold,
            name="block_threshold",
        )

        if self.review_threshold >= self.block_threshold:
            raise ValueError(
                "review_threshold must be strictly less "
                "than block_threshold."
            )

        if not self.version.strip():
            raise ValueError(
                "Decision policy version must not be empty."
            )


@dataclass(frozen=True)
class FraudDecision:
    """Stable output contract for one fraud decision."""

    action: DecisionAction
    calibrated_probability: float
    review_threshold: float
    block_threshold: float
    policy_version: str
    model_version: str
    calibration_method: str
    reason_codes: tuple[ReasonCode, ...]


class RiskDecisionEngine:
    """Convert calibrated fraud risk into an operational action."""

    def __init__(
        self,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self._policy = policy or DecisionPolicy()

    @property
    def policy(self) -> DecisionPolicy:
        """Return the immutable active decision policy."""
        return self._policy

    def decide(
        self,
        score: FraudScore,
        *,
        reason_codes: tuple[ReasonCode, ...] = (),
    ) -> FraudDecision:
        """Produce APPROVE, REVIEW, or BLOCK."""
        probability = float(
            score.calibrated_probability
        )

        _validate_probability(
            probability,
            name="calibrated_probability",
        )

        reasons = _validate_reason_codes(
            reason_codes
        )

        if probability >= self._policy.block_threshold:
            action = DecisionAction.BLOCK
        elif probability >= self._policy.review_threshold:
            action = DecisionAction.REVIEW
        else:
            action = DecisionAction.APPROVE

        return FraudDecision(
            action=action,
            calibrated_probability=probability,
            review_threshold=self._policy.review_threshold,
            block_threshold=self._policy.block_threshold,
            policy_version=self._policy.version,
            model_version=score.model_version,
            calibration_method=score.calibration_method,
            reason_codes=reasons,
        )


def _validate_probability(
    value: float,
    *,
    name: str,
) -> None:
    """Validate one probability-like threshold."""
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a numeric probability."
        )

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric."
        ) from exc

    if (
        not np.isfinite(numeric_value)
        or not 0.0 <= numeric_value <= 1.0
    ):
        raise ValueError(
            f"{name} must be finite and lie in [0, 1]."
        )


def _validate_reason_codes(
    reason_codes: tuple[ReasonCode, ...],
) -> tuple[ReasonCode, ...]:
    """Validate reviewer-facing reasons."""
    if not isinstance(reason_codes, tuple):
        raise TypeError(
            "reason_codes must be a tuple."
        )

    if not all(
        isinstance(reason, ReasonCode)
        for reason in reason_codes
    ):
        raise TypeError(
            "reason_codes must contain only ReasonCode objects."
        )

    return reason_codes