"""HTTP schemas for the FraudShield inference API."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS


class ScoreRequest(BaseModel):
    """One fully engineered transaction for fraud scoring."""

    model_config = ConfigDict(extra="forbid")

    TX_AMOUNT: float

    TX_HOUR: float
    TX_DAY_OF_WEEK: float
    TX_DAY_OF_MONTH: float
    TX_MONTH: float
    TX_IS_WEEKEND: float
    TX_IS_NIGHT: float
    TX_IS_BUSINESS_HOURS: float

    CUSTOMER_TX_COUNT_1H: float
    CUSTOMER_TX_AMOUNT_1H: float
    CUSTOMER_TX_COUNT_24H: float
    CUSTOMER_TX_AMOUNT_24H: float
    CUSTOMER_TX_COUNT_7D: float
    CUSTOMER_TX_AMOUNT_7D: float

    CUSTOMER_HIST_TX_COUNT: float
    CUSTOMER_HIST_MEAN_AMOUNT: float
    CUSTOMER_HIST_MAX_AMOUNT: float
    CUSTOMER_AMOUNT_TO_HIST_MEAN: float

    TERMINAL_HIST_TX_COUNT: float
    TERMINAL_HIST_MEAN_AMOUNT: float
    TERMINAL_AMOUNT_TO_HIST_MEAN: float

    @field_validator("*")
    @classmethod
    def validate_finite(
        cls,
        value: float,
    ) -> float:
        """Reject NaN and infinite model inputs."""
        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValueError(
                "Model features must be finite."
            )

        return numeric

    def to_model_features(
        self,
    ) -> dict[str, float]:
        """Return exact frozen model feature order."""
        payload = self.model_dump()

        return {
            feature: float(payload[feature])
            for feature in MODEL_FEATURE_COLUMNS
        }


class ReasonCodeResponse(BaseModel):
    """Reviewer-facing reason returned by the API."""

    code: str
    message: str
    feature: str
    direction: str
    probability_change: float
    rank: int


class ScoreResponse(BaseModel):
    """Stable public fraud-scoring response."""

    action: str

    raw_probability: float
    calibrated_probability: float

    review_threshold: float
    block_threshold: float

    model_version: str
    calibration_method: str
    policy_version: str

    reason_codes: list[ReasonCodeResponse] = Field(
        default_factory=list
    )


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    service: str