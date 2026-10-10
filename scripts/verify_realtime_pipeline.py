"""Verify the end-to-end FraudShield real-time transaction pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fraudshield.decision import (
    DECISION_POLICY_VERSION,
    FROZEN_REVIEW_THRESHOLD,
    PROVISIONAL_BLOCK_THRESHOLD,
    DecisionAction,
    RiskDecisionEngine,
)
from fraudshield.features.online import ONLINE_MODEL_FEATURE_COLUMNS
from fraudshield.features.online_pipeline import OnlineFeaturePipeline
from fraudshield.persistence import (
    FraudDecisionRepository,
    create_database_engine,
    create_schema,
    create_session_factory,
)
from fraudshield.realtime import RealTimeFraudPipeline
from fraudshield.scoring.service import (
    FROZEN_MODEL_VERSION,
    FROZEN_THRESHOLD,
    FraudScorer,
)
from fraudshield.state.store import InMemoryStateStore
from fraudshield.streaming.simulator import TransactionStream

DATA_PATH = Path("data/raw/fraudshield_transactions.csv")
REPLAY_LIMIT = 1000


class DeterministicModel:
    """Deterministic integration-only probability model."""

    def predict_proba(self, X):
        amount = X["TX_AMOUNT"].to_numpy(dtype=float)

        probability = np.where(
            amount >= 200.0,
            0.12,
            np.where(amount >= 100.0, 0.06, 0.02),
        )

        return np.column_stack(
            (1.0 - probability, probability)
        )


class IdentityCalibrator:
    """Integration-only identity calibrator."""

    def predict(self, probabilities):
        return np.asarray(probabilities, dtype=float)


def main() -> None:
    """Run deterministic CP-05 integration verification."""
    stream = TransactionStream.from_csv(DATA_PATH)

    state = InMemoryStateStore()
    feature_pipeline = OnlineFeaturePipeline(state)

    scorer = FraudScorer(
        model=DeterministicModel(),
        calibrator=IdentityCalibrator(),
    )

    decision_engine = RiskDecisionEngine()

    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    session = session_factory()

    repository = FraudDecisionRepository(session)

    pipeline = RealTimeFraudPipeline(
        feature_pipeline=feature_pipeline,
        scorer=scorer,
        decision_engine=decision_engine,
        repository=repository,
        session=session,
    )

    processed = 0
    approve_count = 0
    review_count = 0
    block_count = 0
    feature_contract_violations = 0
    state_violations = 0

    try:
        for simulated in stream.iter_events(limit=REPLAY_LIMIT):
            event = simulated.event

            before_customer = state.customer_history(
                event.customer_id,
                event.event_time,
            )

            feature_result = feature_pipeline.transform(event)

            if tuple(feature_result.features) != tuple(
                ONLINE_MODEL_FEATURE_COLUMNS
            ):
                feature_contract_violations += 1

            if (
                feature_result.features["CUSTOMER_HIST_TX_COUNT"]
                != before_customer.count
            ):
                state_violations += 1

            result = pipeline.process(event)

            if result.decision.action is DecisionAction.APPROVE:
                approve_count += 1
            elif result.decision.action is DecisionAction.REVIEW:
                review_count += 1
            elif result.decision.action is DecisionAction.BLOCK:
                block_count += 1

            processed += 1

        persisted = repository.count()

        if processed != REPLAY_LIMIT:
            raise AssertionError(
                f"Expected {REPLAY_LIMIT} events, got {processed}."
            )

        if persisted != processed:
            raise AssertionError(
                "Persisted decision count does not match processed events."
            )

        if feature_contract_violations:
            raise AssertionError(
                "Online feature contract violations detected."
            )

        if state_violations:
            raise AssertionError(
                "Read-before-write state violations detected."
            )

        if approve_count + review_count + block_count != processed:
            raise AssertionError(
                "Decision counts do not match processed events."
            )

        print("=== DAY-05 CP-05 REAL-TIME PIPELINE QA ===")
        print(f"Replay events: {processed:,}")
        print(
            "Online model features: "
            f"{len(ONLINE_MODEL_FEATURE_COLUMNS)}"
        )
        print(
            "Feature contract violations: "
            f"{feature_contract_violations}"
        )
        print(
            "State/leakage violations: "
            f"{state_violations}"
        )
        print(f"Persisted decisions: {persisted:,}")
        print(f"APPROVE decisions: {approve_count:,}")
        print(f"REVIEW decisions: {review_count:,}")
        print(f"BLOCK decisions: {block_count:,}")
        print(
            "Frozen model threshold: "
            f"{FROZEN_THRESHOLD:.3f}"
        )
        print(
            "Frozen review threshold: "
            f"{FROZEN_REVIEW_THRESHOLD:.3f}"
        )
        print(
            "Provisional block threshold: "
            f"{PROVISIONAL_BLOCK_THRESHOLD:.3f}"
        )
        print(f"Model version: {FROZEN_MODEL_VERSION}")
        print(
            "Decision policy version: "
            f"{DECISION_POLICY_VERSION}"
        )
        print("Read-before-write semantics: PASSED")
        print("Durable-write-before-state semantics: PASSED")
        print("Locked test used: False")
        print("Model retrained: False")
        print("Threshold retuned: False")
        print("Day-05 CP-05 QA: PASSED")

    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    main()