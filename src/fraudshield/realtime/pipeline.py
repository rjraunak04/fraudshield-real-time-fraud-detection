"""End-to-end real-time fraud transaction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session

from fraudshield.decision import FraudDecision, RiskDecisionEngine
from fraudshield.features.online_pipeline import OnlineFeaturePipeline
from fraudshield.persistence import (
    FraudDecisionRepository,
    FraudDecisionWrite,
)
from fraudshield.scoring.service import FraudScore
from fraudshield.streaming.events import TransactionEvent


class TransactionScorer(Protocol):
    """Minimal scorer contract required by the real-time pipeline."""

    def score_one(
        self,
        features: dict[str, float],
    ) -> FraudScore:
        """Score one model-ready transaction."""


@dataclass(frozen=True, slots=True)
class RealTimeFraudResult:
    """Stable output from one successfully processed transaction."""

    transaction_id: int
    score: FraudScore
    decision: FraudDecision


class RealTimeFraudPipeline:
    """Orchestrate feature, scoring, decision, persistence, and state."""

    def __init__(
        self,
        *,
        feature_pipeline: OnlineFeaturePipeline,
        scorer: TransactionScorer,
        decision_engine: RiskDecisionEngine,
        repository: FraudDecisionRepository,
        session: Session,
    ) -> None:
        self._feature_pipeline = feature_pipeline
        self._scorer = scorer
        self._decision_engine = decision_engine
        self._repository = repository
        self._session = session

    def process(
        self,
        event: TransactionEvent,
    ) -> RealTimeFraudResult:
        """Process one event with durable-write-before-state semantics."""
        feature_result = self._feature_pipeline.transform(event)

        score = self._scorer.score_one(
            feature_result.features
        )

        decision = self._decision_engine.decide(score)

        write = FraudDecisionWrite(
            transaction_id=event.transaction_id,
            event_time=event.event_time,
            customer_id=event.customer_id,
            terminal_id=event.terminal_id,
            amount=float(event.amount),
            raw_probability=score.raw_probability,
            calibrated_probability=score.calibrated_probability,
            decision=decision.action.value,
            model_version=score.model_version,
            policy_version=decision.policy_version,
        )

        try:
            self._repository.add(write)
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        self._feature_pipeline.commit(event)

        return RealTimeFraudResult(
            transaction_id=event.transaction_id,
            score=score,
            decision=decision,
        )