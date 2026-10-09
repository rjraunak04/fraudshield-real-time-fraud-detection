"""FastAPI application for FraudShield inference."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from fraudshield.api.observability import (
    elapsed_ms,
    log_event,
    new_request_id,
)
from fraudshield.api.schemas import (
    HealthResponse,
    RawTransactionRequest,
    RealTimeScoreResponse,
    ReasonCodeResponse,
    ScoreRequest,
    ScoreResponse,
)
from fraudshield.decision.engine import RiskDecisionEngine
from fraudshield.explainability.reason_codes import ReasonCode
from fraudshield.persistence import DuplicateDecisionError
from fraudshield.realtime import (
    RealTimeFraudResult,
)
from fraudshield.scoring.service import FraudScore
from fraudshield.streaming.events import TransactionEvent

logger = logging.getLogger("fraudshield.api")


class ScoringService(Protocol):
    """Protocol required by the engineered-feature HTTP layer."""

    def score_one(
        self,
        transaction: dict[str, float],
    ) -> FraudScore:
        """Score one engineered transaction."""
        ...


class ReasonService(Protocol):
    """Protocol for optional local explanations."""

    def explain(
        self,
        transaction: dict[str, float],
    ) -> tuple[ReasonCode, ...]:
        """Return reviewer-facing reason codes."""
        ...


class RealTimeService(Protocol):
    """Protocol required by the raw-transaction HTTP layer."""

    def process(
        self,
        event: TransactionEvent,
    ) -> RealTimeFraudResult:
        """Process one raw transaction."""
        ...


def _unconfigured_scorer() -> ScoringService:
    """Fail safely until a fitted scorer is injected."""
    raise RuntimeError(
        "FraudShield scoring service is not configured."
    )


def _unconfigured_realtime_service() -> RealTimeService:
    """Fail safely until the real-time pipeline is injected."""
    raise RuntimeError(
        "FraudShield real-time service is not configured."
    )


def create_app(
    *,
    scorer_dependency: Callable[
        [],
        ScoringService,
    ] = _unconfigured_scorer,
    reason_dependency: Callable[
        [],
        ReasonService | None,
    ]
    | None = None,
    decision_engine: RiskDecisionEngine | None = None,
    realtime_dependency: Callable[
        [],
        RealTimeService,
    ] = _unconfigured_realtime_service,
) -> FastAPI:
    """Create the FraudShield HTTP application."""
    app = FastAPI(
        title="FraudShield",
        version="0.1.0",
        description=(
            "Real-time fraud scoring and "
            "risk decision service."
        ),
    )

    engine = decision_engine or RiskDecisionEngine()

    @app.middleware("http")
    async def observability_middleware(
        request: Request,
        call_next,
    ):
        request_id = (
            request.headers.get("X-Request-ID")
            or new_request_id()
        )
        started_at = time.perf_counter()

        log_event(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception:
            latency = elapsed_ms(started_at)

            logger.exception(
                "Unhandled API exception",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                },
            )

            log_event(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                latency_ms=round(latency, 3),
            )

            response = JSONResponse(
                status_code=500,
                content={
                    "detail": (
                        "Internal service error."
                    )
                },
            )

        latency = elapsed_ms(started_at)

        response.headers["X-Request-ID"] = (
            request_id
        )
        response.headers["X-Process-Time-Ms"] = (
            f"{latency:.3f}"
        )

        log_event(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency, 3),
        )

        return response

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    def health() -> HealthResponse:
        """Return API process liveness."""
        return HealthResponse(
            status="ok",
            service="fraudshield",
        )

    @app.get(
        "/ready",
        response_model=HealthResponse,
        tags=["system"],
    )
    def readiness() -> HealthResponse:
        """Return engineered-feature scoring readiness."""
        try:
            scorer_dependency()
        except (
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Fraud scoring service "
                    "is not ready."
                ),
            ) from exc

        return HealthResponse(
            status="ready",
            service="fraudshield",
        )

    @app.get(
        "/ready/realtime",
        response_model=HealthResponse,
        tags=["system"],
    )
    def realtime_readiness() -> HealthResponse:
        """Return raw-transaction pipeline readiness."""
        try:
            realtime_dependency()
        except (
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Fraud real-time service "
                    "is not ready."
                ),
            ) from exc

        return HealthResponse(
            status="ready",
            service="fraudshield-realtime",
        )

    @app.post(
        "/v1/score",
        response_model=ScoreResponse,
        tags=["inference"],
    )
    def score_transaction(
        request: ScoreRequest,
    ) -> ScoreResponse:
        """Score and classify one engineered transaction."""
        features = request.to_model_features()

        try:
            scorer = scorer_dependency()

            score = scorer.score_one(features)

            reasons: tuple[
                ReasonCode,
                ...
            ] = ()

            if reason_dependency is not None:
                reason_service = reason_dependency()

                if reason_service is not None:
                    reasons = reason_service.explain(
                        features
                    )

            decision = engine.decide(
                score,
                reason_codes=reasons,
            )

        except (
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Fraud scoring failed safely."
                ),
            ) from exc

        return ScoreResponse(
            action=decision.action.value,
            raw_probability=score.raw_probability,
            calibrated_probability=(
                score.calibrated_probability
            ),
            review_threshold=(
                decision.review_threshold
            ),
            block_threshold=(
                decision.block_threshold
            ),
            model_version=decision.model_version,
            calibration_method=(
                decision.calibration_method
            ),
            policy_version=(
                decision.policy_version
            ),
            reason_codes=[
                ReasonCodeResponse(
                    code=reason.code,
                    message=reason.message,
                    feature=reason.feature,
                    direction=reason.direction,
                    probability_change=(
                        reason.probability_change
                    ),
                    rank=reason.rank,
                )
                for reason in decision.reason_codes
            ],
        )

    @app.post(
        "/v1/transactions/score",
        response_model=RealTimeScoreResponse,
        tags=["realtime"],
    )
    def score_raw_transaction(
        request: RawTransactionRequest,
    ) -> RealTimeScoreResponse:
        """Process one raw transaction through the real-time pipeline."""
        event = TransactionEvent(
            transaction_id=request.transaction_id,
            event_time=request.event_time,
            customer_id=request.customer_id,
            terminal_id=request.terminal_id,
            amount=request.amount,
        )

        try:
            realtime_service = realtime_dependency()
            result = realtime_service.process(event)

        except DuplicateDecisionError as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Transaction has already been processed."
                ),
            ) from exc

        except (
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Real-time fraud processing failed safely."
                ),
            ) from exc

        score = result.score
        decision = result.decision

        return RealTimeScoreResponse(
            transaction_id=result.transaction_id,
            action=decision.action.value,
            raw_probability=score.raw_probability,
            calibrated_probability=(
                score.calibrated_probability
            ),
            review_threshold=(
                decision.review_threshold
            ),
            block_threshold=(
                decision.block_threshold
            ),
            model_version=decision.model_version,
            calibration_method=(
                decision.calibration_method
            ),
            policy_version=(
                decision.policy_version
            ),
        )

    return app


app = create_app()