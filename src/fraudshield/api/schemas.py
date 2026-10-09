"""Pydantic schemas for FraudShield HTTP API."""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS


class RawTransactionRequest(BaseModel):
    """Raw transaction accepted by the real-time API."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int = Field(ge=0)
    event_time: datetime
    customer_id: int = Field(ge=0)
    terminal_id: int = Field(ge=0)
    amount: float = Field(ge=0.0)

    @field_validator("amount")
    @classmethod
    def validate_amount(
        cls,
        value: float,
    ) -> float:
        """Require a finite transaction amount."""
        if not math.isfinite(value):
            raise ValueError(
                "amount must be finite."
            )

        return value


class ScoreRequest(BaseModel):
    """Engineered model features accepted by scoring API."""

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
        """Require finite engineered feature values."""
        if not math.isfinite(value):
            raise ValueError(
                "All model features must be finite."
            )

        return value

    def to_model_features(
        self,
    ) -> dict[str, float]:
        """Return features in canonical model order."""
        payload = self.model_dump()

        return {
            column: float(payload[column])
            for column in MODEL_FEATURE_COLUMNS
        }


class ReasonCodeResponse(BaseModel):
    """One human-readable fraud reason code."""

    code: str
    message: str
    feature: str
    direction: str
    probability_change: float
    rank: int


class ScoreResponse(BaseModel):
    """Response from engineered-feature scoring API."""

    action: str
    raw_probability: float
    calibrated_probability: float
    review_threshold: float
    block_threshold: float
    model_version: str
    calibration_method: str
    policy_version: str
    reason_codes: list[ReasonCodeResponse]


class RealTimeScoreResponse(BaseModel):
    """Response from raw-transaction real-time API."""

    transaction_id: int
    action: str
    raw_probability: float
    calibrated_probability: float
    review_threshold: float
    block_threshold: float
    model_version: str
    calibration_method: str
    policy_version: str


class HealthResponse(BaseModel):
    """Liveness or readiness response."""

    status: str
    service: str


class OperationalMetricsResponse(BaseModel):
    """Dashboard-ready operational aggregates."""

    total_transactions: int

    approve_count: int
    review_count: int
    block_count: int

    approve_rate: float
    review_rate: float
    block_rate: float

    average_calibrated_probability: float
    high_risk_rate: float

    total_transaction_amount: float
    average_transaction_amount: float


class RecentDecisionResponse(BaseModel):
    """One recent fraud decision for dashboard display."""

    transaction_id: int
    event_time: datetime
    customer_id: int
    terminal_id: int
    amount: float
    calibrated_probability: float
    decision: str
    model_version: str
    policy_version: str


class OperationalDashboardResponse(BaseModel):
    """Combined operational dashboard payload."""

    metrics: OperationalMetricsResponse
    recent_decisions: list[RecentDecisionResponse]