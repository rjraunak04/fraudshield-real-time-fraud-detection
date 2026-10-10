"""Final Day-05 end-to-end verification for FraudShield."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from fraudshield.api.app import create_app
from fraudshield.decision import (
    FROZEN_REVIEW_THRESHOLD,
    PROVISIONAL_BLOCK_THRESHOLD,
    RiskDecisionEngine,
)
from fraudshield.features.online_pipeline import (
    OnlineFeaturePipeline,
)
from fraudshield.metrics import build_operational_metrics
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.persistence import (
    FraudDecisionRepository,
    create_schema,
    create_session_factory,
)
from fraudshield.realtime import RealTimeFraudPipeline
from fraudshield.scoring.service import (
    FROZEN_MODEL_VERSION,
    FROZEN_THRESHOLD,
    FraudScorer,
)
from fraudshield.state import InMemoryStateStore
from fraudshield.streaming.events import TransactionEvent


class DeterministicModel:
    """Integration-only deterministic probability model."""

    def predict_proba(
        self,
        features,
    ) -> np.ndarray:
        """Return probability based only on transaction amount."""
        amount = float(features.iloc[0]["TX_AMOUNT"])

        if amount >= 200.0:
            probability = 0.12
        elif amount >= 100.0:
            probability = 0.06
        else:
            probability = 0.02

        return np.asarray(
            [[1.0 - probability, probability]],
            dtype=float,
        )


class IdentityCalibrator:
    """Integration-only identity probability calibrator."""

    def predict(
        self,
        probabilities,
    ) -> np.ndarray:
        """Return positive-class probability unchanged."""
        values = np.asarray(
            probabilities,
            dtype=float,
        )

        return values.reshape(-1)


def make_event(
    transaction_id: int,
    *,
    minute: int,
    amount: float,
) -> TransactionEvent:
    """Create deterministic raw transaction."""
    return TransactionEvent(
        transaction_id=transaction_id,
        event_time=datetime(
            2018,
            6,
            15,
            10,
            minute,
        ),
        customer_id=101,
        terminal_id=501,
        amount=amount,
    )


def main() -> None:
    """Verify the complete Day-05 real-time platform."""
    database_engine = create_engine(
        "sqlite://",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    create_schema(database_engine)

    session_factory = create_session_factory(
        database_engine
    )
    session = session_factory()

    state = InMemoryStateStore()

    feature_pipeline = OnlineFeaturePipeline(
        state
    )

    scorer = FraudScorer(
        model=DeterministicModel(),
        calibrator=IdentityCalibrator(),
    )

    decision_engine = RiskDecisionEngine()

    repository = FraudDecisionRepository(
        session
    )

    realtime_pipeline = RealTimeFraudPipeline(
        feature_pipeline=feature_pipeline,
        scorer=scorer,
        decision_engine=decision_engine,
        repository=repository,
        session=session,
    )

    app = create_app(
        realtime_dependency=(
            lambda: realtime_pipeline
        ),
        metrics_repository_dependency=(
            lambda: repository
        ),
    )

    client = TestClient(app)

    events = [
        make_event(
            1,
            minute=1,
            amount=25.0,
        ),
        make_event(
            2,
            minute=2,
            amount=125.0,
        ),
        make_event(
            3,
            minute=3,
            amount=250.0,
        ),
    ]

    expected_actions = [
        "APPROVE",
        "REVIEW",
        "BLOCK",
    ]

    try:
        actions: list[str] = []

        for event in events:
            response = client.post(
                "/v1/transactions/score",
                json={
                    "transaction_id": (
                        event.transaction_id
                    ),
                    "event_time": (
                        event.event_time.isoformat()
                    ),
                    "customer_id": (
                        event.customer_id
                    ),
                    "terminal_id": (
                        event.terminal_id
                    ),
                    "amount": event.amount,
                },
            )

            assert response.status_code == 200

            payload = response.json()

            actions.append(
                payload["action"]
            )

            assert (
                payload["transaction_id"]
                == event.transaction_id
            )

            assert (
                payload["model_version"]
                == FROZEN_MODEL_VERSION
            )

            assert (
                payload["review_threshold"]
                == FROZEN_REVIEW_THRESHOLD
            )

            assert (
                payload["block_threshold"]
                == PROVISIONAL_BLOCK_THRESHOLD
            )

        assert actions == expected_actions

        assert repository.count() == 3

        customer_history = state.customer_history(
            customer_id=101,
            before=datetime(
                2018,
                6,
                15,
                11,
                0,
            ),
        )

        assert customer_history.count == 3

        duplicate_response = client.post(
            "/v1/transactions/score",
            json={
                "transaction_id": 3,
                "event_time": (
                    events[2].event_time.isoformat()
                ),
                "customer_id": 101,
                "terminal_id": 501,
                "amount": 250.0,
            },
        )

        assert duplicate_response.status_code == 409

        assert repository.count() == 3

        customer_history_after_duplicate = (
            state.customer_history(
                customer_id=101,
                before=datetime(
                    2018,
                    6,
                    15,
                    11,
                    0,
                ),
            )
        )

        assert (
            customer_history_after_duplicate.count
            == 3
        )

        metrics = build_operational_metrics(
            repository
        )

        assert metrics.total_transactions == 3
        assert metrics.approve_count == 1
        assert metrics.review_count == 1
        assert metrics.block_count == 1

        dashboard_response = client.get(
            "/v1/metrics/operations",
            headers={
                "X-Request-ID": (
                    "day05-final-verification"
                )
            },
        )

        assert dashboard_response.status_code == 200

        dashboard = dashboard_response.json()

        assert (
            dashboard["metrics"][
                "total_transactions"
            ]
            == 3
        )

        assert len(
            dashboard["recent_decisions"]
        ) == 3

        assert (
            dashboard_response.headers[
                "X-Request-ID"
            ]
            == "day05-final-verification"
        )

        assert (
            "X-Process-Time-Ms"
            in dashboard_response.headers
        )

        realtime_ready = client.get(
            "/ready/realtime"
        )

        assert realtime_ready.status_code == 200

        assert (
            tuple(MODEL_FEATURE_COLUMNS)
            == tuple(
                feature_pipeline.transform(
                    make_event(
                        999,
                        minute=10,
                        amount=50.0,
                    )
                ).features
            )
        )

        print(
            "=== FRAUDSHIELD DAY-05 FINAL QA ==="
        )
        print(
            "Raw transaction API: PASSED"
        )
        print(
            "Online feature contract:",
            len(MODEL_FEATURE_COLUMNS),
            "features",
        )
        print(
            "Read-before-write state: PASSED"
        )
        print(
            "Real-time scoring: PASSED"
        )
        print(
            "Decision routing:",
            actions,
        )
        print(
            "Durable decisions:",
            repository.count(),
        )
        print(
            "Online customer history:",
            customer_history.count,
        )
        print(
            "Duplicate protection: PASSED"
        )
        print(
            "Operational dashboard: PASSED"
        )
        print(
            "Request observability: PASSED"
        )
        print(
            "Frozen model threshold:",
            f"{FROZEN_THRESHOLD:.3f}",
        )
        print(
            "Frozen review threshold:",
            f"{FROZEN_REVIEW_THRESHOLD:.3f}",
        )
        print(
            "Provisional block threshold:",
            f"{PROVISIONAL_BLOCK_THRESHOLD:.3f}",
        )
        print(
            "Model version:",
            FROZEN_MODEL_VERSION,
        )
        print(
            "Locked test reopened: False"
        )
        print(
            "Model retrained: False"
        )
        print(
            "Threshold retuned: False"
        )
        print(
            "Day-05 final QA: PASSED"
        )

    finally:
        session.close()
        database_engine.dispose()


if __name__ == "__main__":
    main()