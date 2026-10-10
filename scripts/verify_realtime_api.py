"""Verify Day-05 CP-06 real-time FraudShield API integration."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from fraudshield.api.app import create_app
from fraudshield.decision import (
    DECISION_POLICY_VERSION,
    FROZEN_REVIEW_THRESHOLD,
    PROVISIONAL_BLOCK_THRESHOLD,
    RiskDecisionEngine,
)
from fraudshield.features.online_pipeline import OnlineFeaturePipeline
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
from fraudshield.state.store import InMemoryStateStore


class VerificationModel:
    """Deterministic integration-only verification model."""

    def predict_proba(self, X):
        amount = float(X.iloc[0]["TX_AMOUNT"])

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
    """Return raw probabilities unchanged."""

    def predict(self, probabilities):
        return np.asarray(probabilities, dtype=float)


def main() -> None:
    """Run deterministic CP-06 API verification."""
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
        model=VerificationModel(),
        calibrator=IdentityCalibrator(),
    )

    repository = FraudDecisionRepository(
        session
    )

    pipeline = RealTimeFraudPipeline(
        feature_pipeline=feature_pipeline,
        scorer=scorer,
        decision_engine=RiskDecisionEngine(),
        repository=repository,
        session=session,
    )

    app = create_app(
        realtime_dependency=lambda: pipeline
    )
    client = TestClient(app)

    transactions = [
        {
            "transaction_id": 900001,
            "event_time": "2018-06-15T10:00:00",
            "customer_id": 101,
            "terminal_id": 501,
            "amount": 25.0,
        },
        {
            "transaction_id": 900002,
            "event_time": "2018-06-15T10:05:00",
            "customer_id": 101,
            "terminal_id": 502,
            "amount": 125.0,
        },
        {
            "transaction_id": 900003,
            "event_time": "2018-06-15T10:10:00",
            "customer_id": 101,
            "terminal_id": 503,
            "amount": 250.0,
        },
    ]

    responses = []

    for payload in transactions:
        response = client.post(
            "/v1/transactions/score",
            json=payload,
            headers={
                "X-Request-ID": (
                    f"cp06-{payload['transaction_id']}"
                )
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                "Real-time API verification request failed: "
                f"{response.status_code} {response.text}"
            )

        responses.append(response)

    actions = [
        response.json()["action"]
        for response in responses
    ]

    expected_actions = [
        "APPROVE",
        "REVIEW",
        "BLOCK",
    ]

    if actions != expected_actions:
        raise AssertionError(
            f"Unexpected actions: {actions}"
        )

    if repository.count() != 3:
        raise AssertionError(
            "Expected exactly three persisted decisions."
        )

    customer_history = state.customer_history(
        101,
        datetime(2018, 6, 16),
    )

    if customer_history.count != 3:
        raise AssertionError(
            "Expected exactly three committed "
            "customer transactions."
        )

    duplicate = client.post(
        "/v1/transactions/score",
        json=transactions[0],
    )

    if duplicate.status_code != 409:
        raise AssertionError(
            "Duplicate transaction did not return HTTP 409."
        )

    invalid = client.post(
        "/v1/transactions/score",
        json={
            **transactions[0],
            "transaction_id": -1,
        },
    )

    if invalid.status_code != 422:
        raise AssertionError(
            "Invalid transaction did not return HTTP 422."
        )

    readiness = client.get(
        "/ready/realtime"
    )

    if readiness.status_code != 200:
        raise AssertionError(
            "Real-time readiness endpoint failed."
        )

    first_request_id = responses[0].headers.get(
        "X-Request-ID"
    )

    if first_request_id != "cp06-900001":
        raise AssertionError(
            "Request ID was not propagated."
        )

    if "X-Process-Time-Ms" not in responses[0].headers:
        raise AssertionError(
            "Process-time header is missing."
        )

    print(
        "=== DAY-05 CP-06 REAL-TIME API QA ==="
    )
    print(
        "Raw transaction endpoint: "
        "POST /v1/transactions/score"
    )
    print(
        "Successful requests: 3"
    )
    print(
        f"Decision routing: {actions}"
    )
    print(
        f"Persisted decisions: {repository.count()}"
    )
    print(
        "Committed customer history: "
        f"{customer_history.count}"
    )
    print(
        "Duplicate transaction HTTP status: "
        f"{duplicate.status_code}"
    )
    print(
        "Invalid transaction HTTP status: "
        f"{invalid.status_code}"
    )
    print(
        "Real-time readiness: PASSED"
    )
    print(
        "Request-ID propagation: PASSED"
    )
    print(
        "Process-time observability: PASSED"
    )
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
    print(
        f"Model version: {FROZEN_MODEL_VERSION}"
    )
    print(
        "Decision policy version: "
        f"{DECISION_POLICY_VERSION}"
    )
    print(
        "Locked test used: False"
    )
    print(
        "Model retrained: False"
    )
    print(
        "Threshold retuned: False"
    )
    print(
        "Day-05 CP-06 QA: PASSED"
    )

    session.close()
    database_engine.dispose()


if __name__ == "__main__":
    main()